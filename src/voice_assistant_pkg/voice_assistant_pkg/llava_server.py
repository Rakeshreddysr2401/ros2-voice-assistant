#!/usr/bin/env python3
import os
import cv2
import threading
import time
from PIL import Image
import torch
import rclpy
from rclpy.node import Node
from custom_interfaces.msg import VisualQuery


class LLaVAServer(Node):
    def __init__(self):
        super().__init__("llava_server")

        # Camera setup
        camera_source = os.getenv("CAMERA_SOURCE", "0")
        if camera_source.isdigit():
            camera_source = int(camera_source)

        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ Failed to open camera {camera_source}")
            exit(1)

        # Load LLaVA model
        self.load_llava_model()

        # Frame buffer
        self.latest_frame = None
        self.lock = threading.Lock()

        # Start camera thread
        self.running = True
        self.thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.thread.start()

        # Create subscriber for queries and publisher for responses
        self.query_sub = self.create_subscription(VisualQuery, "visual_query_request", self.handle_query, 10)
        self.response_pub = self.create_publisher(VisualQuery, "visual_query_response", 10)
        self.get_logger().info("🦙 LLaVA Server ready (topics: visual_query_request/response)")

    def load_llava_model(self):
        """Load LLaVA model with fallback options"""
        try:
            # Try LLaVA-1.5 first (best performance)
            self.get_logger().info("Loading LLaVA-1.5-7B model...")
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.eval.run_llava import eval_model

            model_path = "liuhaotian/llava-v1.5-7b"

            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=get_model_name_from_path(model_path)
            )

            self.model_type = "llava-1.5"
            self.get_logger().info("✅ LLaVA-1.5 loaded successfully!")

        except Exception as e:
            self.get_logger().warn(f"LLaVA-1.5 failed ({e}), trying LLaVA-1.6...")

            try:
                # Try LLaVA-1.6 (newer version)
                from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

                self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-v1.6-mistral-7b-hf",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )

                self.model_type = "llava-1.6"
                self.get_logger().info("✅ LLaVA-1.6 loaded successfully!")

            except Exception as e2:
                self.get_logger().error(f"Both LLaVA versions failed. Falling back to BLIP-2...")
                self.load_blip_fallback()

    def load_blip_fallback(self):
        """Fallback to BLIP-2 if LLaVA fails"""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16
        )
        self.model_type = "blip2-fallback"
        self.get_logger().info("✅ BLIP-2 fallback loaded!")

    def camera_loop(self):
        """Continuous camera capture"""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame.copy()
            time.sleep(0.05)

    def handle_query(self, request):
        """Process visual queries with LLaVA"""
        frame = None
        with self.lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        response_msg = VisualQuery()
        response_msg.query = request.query

        if frame is None:
            response_msg.response = "⚠️ No camera frame available"
            self.response_pub.publish(response_msg)
            return

        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            if self.model_type == "llava-1.5":
                result = self.process_llava_15(image, request.query)
            elif self.model_type == "llava-1.6":
                result = self.process_llava_16(image, request.query)
            else:  # BLIP-2 fallback
                result = self.process_blip_fallback(image, request.query)

            self.get_logger().info(f"Query: '{request.query}' -> Response: '{result}'")

        except Exception as e:
            result = f"Error processing: {str(e)}"
            self.get_logger().error(f"Processing error: {e}")

        response_msg.response = result
        self.response_pub.publish(response_msg)

    def process_llava_15(self, image, query):
        """Process with LLaVA-1.5"""
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        if not query.strip():
            prompt = "Describe this image in detail."
        else:
            prompt = query

        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_formatted, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors='pt').unsqueeze(0)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                use_cache=True
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        return outputs

    def process_llava_16(self, image, query):
        """Process with LLaVA-1.6"""
        if not query.strip():
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        else:
            prompt = f"USER: <image>\n{query}\nASSISTANT:"

        inputs = self.processor(prompt, image, return_tensors="pt")

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

        result = self.processor.decode(output[0], skip_special_tokens=True)
        # Clean the response
        result = result.split("ASSISTANT:")[-1].strip()
        return result

    def process_blip_fallback(self, image, query):
        """Process with BLIP-2 fallback"""
        if not query.strip():
            inputs = self.processor(images=image, return_tensors="pt")
            generated_ids = self.model.generate(**inputs, max_length=50)
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        else:
            prompt = f"Question: {query} Answer:"
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            generated_ids = self.model.generate(**inputs, max_length=30)
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            if result.startswith(prompt):
                result = result[len(prompt):].strip()
        return result

    def destroy_node(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LLaVAServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
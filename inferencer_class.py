from bespokelabs import curator
from PIL import Image
import os
import re
import io
import base64


def get_mime_type_from_image(image_path):

    try:
        with Image.open(image_path) as img:
            img_format = img.format.lower() if img.format else None
            if img_format:
                return f"image/{img_format}"
        return None
    except Exception as e:
        return None


class NextTurnGenerator(curator.LLM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def prompt(self, example):

        def _get_local_image_uri(image_input):
            # Handle both file paths and PIL Image objects
            if isinstance(image_input, str):
                # If it's a file path string
                img = curator.types.Image(url=image_input)
            elif hasattr(image_input, 'save'):  # Check if it's a PIL Image object
                # If it's a PIL Image object, we need to save it temporarily or convert it                
                # Convert PIL Image to base64
                buffer = io.BytesIO()
                image_input.save(buffer, format='PNG')
                img_data = buffer.getvalue()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                img_uri = f"data:image/png;base64,{img_base64}"
                return img_uri
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            img_uri = f"data:{img.mime_type};base64,{img.serialize()}"
            return img_uri

        def _process_content(content):
            """Recursively process content to convert image paths to base64 URIs"""
            processed_content = []
            for item in content:
                processed_item = {}
                for key, value in item.items():
                    if key == "image_url" and isinstance(value, dict):
                        image_input = value['url']
                        processed_item[key] = {"url": _get_local_image_uri(image_input)}
                    else:
                        processed_item[key] = value
                processed_content.append(processed_item)
            return processed_content

        # Process all messages to convert image paths
        messages = example['messages']
        processed_messages = []
        
        for message in messages:
            processed_message = message.copy()
            processed_message['content'] = _process_content(processed_message['content'])
            processed_messages.append(processed_message)

        return processed_messages

    
    def parse(self, example, response):

        example['response'] = response

        return example
import os
import json
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from biz.llm.client.base import BaseClient
from biz.llm.types import NotGiven, NOT_GIVEN
from biz.utils.log import logger


class BedrockClient(BaseClient):
    def __init__(self):
        self.default_model = os.getenv("BEDROCK_API_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0")
        
        # 优先使用默认的环境中的role授权，如果没有再使用环境变量的aws access key secret key
        try:
            # 尝试使用默认凭证（IAM role）
            self.client = boto3.client('bedrock-runtime')
        except NoCredentialsError:
            # 如果默认凭证失败，使用环境变量
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            
            if not aws_access_key or not aws_secret_key:
                raise ValueError("AWS credentials are required. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables or configure IAM role.")
            
            self.client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )

    def completions(self,
                    messages: List[Dict[str, str]],
                    model: Optional[str] | NotGiven = NOT_GIVEN,
                    ) -> str:
        try:
            model = model or self.default_model
            logger.debug(f"Sending request to Bedrock API. Model: {model}, Messages: {messages}")
            
            # 使用Messages API格式
            bedrock_messages = self._convert_to_bedrock_messages(messages)
            
            body = json.dumps({
                "messages": bedrock_messages["messages"],
                "max_tokens": 4000,
                "temperature": 0.7,
                "top_p": 0.9,
                "anthropic_version": "bedrock-2023-05-31"
            })
            
            # 如果有system消息，添加到body中
            if bedrock_messages["system"]:
                body_dict = json.loads(body)
                body_dict["system"] = bedrock_messages["system"]
                body = json.dumps(body_dict)
            
            response = self.client.invoke_model(
                body=body,
                modelId=model,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            
            if 'content' in response_body and response_body['content']:
                return response_body['content'][0]['text'].strip()
            else:
                logger.error("Unexpected response format from Bedrock API")
                return "AI服务返回格式异常，请稍后重试"
                
        except ClientError as e:
            logger.error(f"Bedrock API error: {str(e)}")
            error_code = e.response['Error']['Code']
            if error_code == 'UnauthorizedOperation':
                return "Bedrock API认证失败，请检查AWS凭证是否正确"
            elif error_code == 'ValidationException':
                return "Bedrock API请求参数错误，请检查模型ID是否正确"
            else:
                return f"调用Bedrock API时出错: {str(e)}"
        except Exception as e:
            logger.error(f"Bedrock API error: {str(e)}")
            return f"调用Bedrock API时出错: {str(e)}"

    def _convert_to_bedrock_messages(self, messages: List[Dict[str, str]]) -> Dict:
        """将OpenAI格式的消息转换为Bedrock Messages API格式"""
        bedrock_messages = []
        system_message = None
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                system_message = content
            elif role == "user":
                bedrock_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": content}]
                })
            elif role == "assistant":
                bedrock_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}]
                })
        
        return {
            "messages": bedrock_messages,
            "system": system_message
        }

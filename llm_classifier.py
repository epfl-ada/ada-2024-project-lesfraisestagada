from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import gc
import os
import threading


class Generator:
    def __init__(self, local_compute = True, model_key = None, model_family = None):
        self.local_compute = local_compute
        self.model_key = model_key
        self.model_family = model_family
        
        if self.local_compute:
            try:
                import torch
                
                self.torch = torch
            except ImportError:
                raise ImportError("torch is not installed. Please install torch to use local compute")
        else:
            self.torch = None
        
        self.chat_template = {
            "llama" : "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n",
            "qwen" : "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'Please reason step by step, and put your final answer within \\\\boxed{}.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nPlease reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n",
        }
        
        if self.local_compute:
            self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        else:
            self.quantization_config = None
        self.loaded_models = {}
        
        self.hf_token = os.getenv("HF_TOKEN")
        
        if self.model_key is None or self.model_family is None:
            raise ValueError("Please provide a model key and a model family.")
    
    
    def load_tokenizer(self):
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_key, token=self.hf_token)
            
        return tokenizer

    def load_model(self):
        if self.model_key in self.loaded_models:
            print(f"Model {self.model_key} already loaded, reusing.")
            return self.loaded_models[self.model_key]
    
            
        if self.model_family in ["llama"]:
            model = LlamaForCausalLM.from_pretrained(self.model_key,
                                                     quantization_config=self.quantization_config,
                                                     low_cpu_mem_usage=True,
                                                     torch_dtype="auto",
                                                     device_map="auto",
                                                     token=self.hf_token
                                                     )
            
        elif self.model_family in ["qwen"]:
            model = AutoModelForCausalLM.from_pretrained(self.model_key,
                                                         quantization_config=self.quantization_config,
                                                         low_cpu_mem_usage=True,
                                                         torch_dtype="auto",
                                                         device_map="auto"
                                                         )
        
        tokenizer = self.load_tokenizer()
        self.loaded_models[self.model_key] = (model, tokenizer)
        print(f"Model {self.model_key} loaded on device {model.device}")
        return model, tokenizer
    
    
    def chat_to_dict(self, inputs):
        if type(inputs) == str:
            inputs = [{"role" : "user", "content" : inputs}]
            
        return inputs
    
    
    def get_model(self, model_key):
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        else:
            return self.load_model(model_key)
            
    
    def add_system_prompt(self, system_prompt, chat):
        if chat[0]["role"] != "system":
            chat = [{"role" : "system", "content" : system_prompt}] + chat
            
        return chat
    
    
    def apply_chat_template(self, chat, tokenizer, show_template = False):
                  
        tokenizer.chat_template = self.chat_template[self.model_family]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if show_template:
            print(f"Prompt: {prompt}")

        return prompt
    
    
    def generate(self, chat, max_new_tokens=512):
        """This function is only used for the generate and should not be used for the pipeline implementation
        """
        
        model, tokenizer = self.get_model(self.model_key)

        inputs = tokenizer.encode(chat, add_special_tokens=True, return_tensors="pt").to(model.device)
        input_length = inputs.shape[-1]
        
        with self.torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
            
        generated_ids = outputs[:, input_length:]
        
        del inputs
        del outputs
        gc.collect()
        self.torch.cuda.empty_cache()
        
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    
    
    def streamed_generate(self, chat, max_new_tokens=512):
             
        model, tokenizer = self.get_model(self.model_key)
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                        
        inputs = tokenizer.encode(chat, add_special_tokens=True, return_tensors="pt").to(model.device)
        
        generation_thread = threading.Thread(
            target= model.generate, 
            args= (inputs,), 
            kwargs={'streamer': streamer, 'max_new_tokens': max_new_tokens},
        )
        
        with self.torch.no_grad():
            generation_thread.start()
            
        return streamer

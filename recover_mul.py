import json

with open('/home/illia/.gemini/antigravity/brain/d7390ae8-0b33-458b-b4d7-bf3d98c59286/.system_generated/logs/transcript_full.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'PLANNER_RESPONSE':
            for tool_call in data.get('tool_calls', []):
                if tool_call['name'] in ['multi_replace_file_content', 'replace_file_content', 'write_to_file']:
                    chunks = tool_call['args'].get('ReplacementChunks', [])
                    if tool_call['name'] == 'replace_file_content':
                        chunks = [tool_call['args']]
                    elif tool_call['name'] == 'write_to_file':
                        chunks = [{'ReplacementContent': tool_call['args'].get('CodeContent', '')}]
                    
                    for chunk in chunks:
                        content = chunk.get('ReplacementContent', '')
                        if 'struct MulLoweringToLinalg' in content:
                            with open('mul_pass.cpp', 'w') as out:
                                out.write(content)

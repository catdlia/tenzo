import json

bit_elut = None
bit_tl1 = None

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
                        if 'struct BitLinearElutLoweringToLinalg' in content:
                            bit_elut = content
                        if 'struct BitLinearTL1LoweringToLinalg' in content:
                            bit_tl1 = content

if bit_elut:
    with open('bit_elut.cpp', 'w') as out: out.write(bit_elut)
if bit_tl1:
    with open('bit_tl1.cpp', 'w') as out: out.write(bit_tl1)

import ast

def extract_functions(source_file, target_file, function_names):
    with open(source_file, 'r', encoding='utf-8') as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    
    extracted_code = []
    
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            # Get the exact source lines for this AST node
            start_lineno = node.lineno - 1
            # For decorators we'd need node.decorator_list[0].lineno
            # node.end_lineno is available in python 3.8+
            end_lineno = node.end_lineno
            
            lines = source_code.split('\n')[start_lineno:end_lineno]
            extracted_code.append('\n'.join(lines))
            
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(extracted_code))
        f.write("\n")

if __name__ == "__main__":
    extract_functions('worldcup_predict.py', 'prompt_builder.py', ['_normalize_bookmakers_payload', 'build_prompt_odds_block'])

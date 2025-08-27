"""
Script to automatically add conditional logic to all baseline sections.
"""

def fix_baseline_conditionals():
    """Add conditional logic to baseline_evaluation.py for selective baseline training."""
    
    # Read the current file
    with open('baseline_evaluation.py', 'r') as f:
        content = f.read()
    
    # Define the baseline sections to wrap
    baseline_sections = [
        ('neural_process', '# 2. Evaluate Neural Process Baseline', '# 3. Evaluate CLMU Baseline'),
        ('clmu', '# 3. Evaluate CLMU Baseline', '# 4. Evaluate State-Space Model Baseline'),
        ('state_space', '# 4. Evaluate State-Space Model Baseline', '# 5. Evaluate existing TimeSeriesTransformer models'),
        ('transformers', '# 5. Evaluate existing TimeSeriesTransformer models', '# 6. Summary comparison')
    ]
    
    # Process each section
    for baseline_name, start_marker, end_marker in baseline_sections:
        print(f"Processing {baseline_name} section...")
        
        # Find the start and end positions
        start_pos = content.find(start_marker)
        if start_pos == -1:
            print(f"Warning: Could not find start marker for {baseline_name}")
            continue
            
        # Handle the end marker - if it's the summary, find a different pattern
        if end_marker == '# 6. Summary comparison':
            end_pos = content.find('# 4. Summary comparison')  # Updated section number
        else:
            end_pos = content.find(end_marker)
            
        if end_pos == -1:
            print(f"Warning: Could not find end marker for {baseline_name}")
            continue
        
        # Extract the section
        section = content[start_pos:end_pos]
        
        # Wrap the section content (excluding the comment line)
        lines = section.split('\n')
        comment_line = lines[0]
        section_content = '\n'.join(lines[1:])
        
        # Create the wrapped section
        if baseline_name == 'transformers':
            wrapped_section = f"""{comment_line}
    if 'transformers' in baselines_to_run:
{_indent_content(section_content, 2)}
    else:
        print("Skipping Transformer Baselines (not in selected baselines)")
"""
        else:
            wrapped_section = f"""{comment_line}
    if '{baseline_name}' in baselines_to_run:
{_indent_content(section_content, 2)}
    else:
        print("Skipping {baseline_name.replace('_', ' ').title()} Baseline (not in selected baselines)")
"""
        
        # Replace the section in the content
        content = content[:start_pos] + wrapped_section + content[end_pos:]
    
    # Write the modified content back
    with open('baseline_evaluation.py', 'w') as f:
        f.write(content)
    
    print("✅ Successfully added conditional logic to all baseline sections!")

def _indent_content(content: str, spaces: int) -> str:
    """Indent all lines in content by the specified number of spaces."""
    indent = ' ' * spaces
    lines = content.split('\n')
    indented_lines = []
    for line in lines:
        if line.strip():  # Only indent non-empty lines
            indented_lines.append(indent + line)
        else:
            indented_lines.append(line)
    return '\n'.join(indented_lines)

if __name__ == "__main__":
    fix_baseline_conditionals()
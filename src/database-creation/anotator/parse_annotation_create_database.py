import json
import re

# pattern_list = [[r'\s?\[dblookup\(([^,]+),\s*([^,]+)\)\s*->\s*(.*?)\]',
#                        r"\s?\[dblookup\('(.+?)',\s*'(.+?)'\) ->\s*(.+?)\]"]]

def parse_db_lookups(file_path):
    """
    Parse a JSON file containing strings and extract database lookup tokens.
    """
    # Read and parse JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Regular expression to match dblookup tokens
    pattern = r'\s?\[dblookup\(([^,]+),\s*([^,]+)\)\s*->\s*(.*?)\]'
    
    results = {"entities" : [], "relationships" : [], "return_values" : [],  "triplets" : []}
    
    # Process each string in the list
    for text in data:
        matches = re.findall(pattern, text)
        for match in matches:
            entity = match[0]
            relationship = match[1] 
            value = match[2]
            results["entities"].append(entity)
            results["relationships"].append(relationship)
            results["return_values"].append(value)
            results["triplets"].append([
                entity,
                relationship,
                value
            ])
            
            # Print extracted information
            print(f"Entity: {entity}")
            print(f"Relationship: {relationship}")
            print(f"Value: {value}")
            print("-" * 40)
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace 'input.json' with your actual file path
    extracted_data = parse_db_lookups('/home/rtn27/LMLM/build-database/annotation/annotated_results.json')
    # Save extracted data to JSON file
    output_path = '/home/rtn27/LMLM/build-database/triplets/hotpotqa_1k_42_dev_triplets.json'
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(extracted_data, output_file, indent=2, ensure_ascii=False)
        
    print(f"Extracted data saved to {output_path}")
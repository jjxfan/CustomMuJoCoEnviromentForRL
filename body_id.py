import xml.etree.ElementTree as ET

def get_body_ids(xml_path):
    """
    Parses a MuJoCo XML file and retrieves body names and their IDs.
    
    Parameters:
    xml_path (str): Path to the MuJoCo XML file
    
    Returns:
    dict: A dictionary with body names as keys and their IDs as values
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    body_ids = {}
    for i, body in enumerate(root.findall('.//body')):
        name = body.get('name')
        if name:
            body_ids[name] = i

    return body_ids

# Path to your MuJoCo XML file
xml_path = "assets/ramp.xml"

# Retrieve body names and their corresponding IDs
body_ids = get_body_ids(xml_path)

# Print body names and their IDs
for name, body_id in body_ids.items():
    print(f"Body Name: {name}, Body ID: {body_id}")
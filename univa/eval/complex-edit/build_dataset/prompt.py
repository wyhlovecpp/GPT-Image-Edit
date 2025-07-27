EDIT_CATEGORIES = [
    (
        "Object Manipulation and Transformation", [
            ("Add an Object", "Insert a new element into the image."),
            ("Remove an Object", "Eliminate an existing element from the image."),
            ("Replace an Object", "Swap one element with another."),
            ("Move an Object", "Change the position of an existing element within the image."),
            ("Resize an Object", "Adjust the size of an existing element."),
            ("Rotate an Object", "Rotate an element to a specified angle."),
            ("Duplicate an Object", "Create a copy of an existing element."),
        ],
    ),
    (
        "Color and Tone Adjustments", [
            ("Change Color", "Replace the color of an element with a specified color."),
            ("Apply Filter/Weather", "Add a color filter or weather effect to the entire image or specific parts."),
        ],
    ),
    (
        "Texture and Material Adjustments", [
            ("Change Texture", "Apply a texture to an element (e.g., change from metal to wood)."),
        ],
    ),
    (
        "Background and Environment", [
            ("Change Background", "Replace the background with a different scene or color."),
        ],
    ),
    (
        "Lighting and Shadows", [
            ("Adjust Lighting", "Change the overall lighting or lighting of specific elements."),
        ],
    ),
    (
        "Text and Symbols", [
            ("Add Text", "Insert text into the image."),
            ("Remove Text", "Eliminate existing text from the image."),
            ("Change Text Properties", "Modify font, color, size, or position of existing text."),
        ],
    ),
    (
        "Pose and Expression", [
            ("Change Pose", "Modify the stance or posture of a person or object."),
            ("Change Facial Expression", "Alter the facial expression of a character."),
        ],
    ),
    (
        "Composition and Cropping", [
            ("Crop Image", "Adjust the framing of the image by removing outer areas."),
            ("Reframe Composition", "Change the focus or arrangement of elements within the image."),
            ("Zoom In/Out", "Adjust the zoom level to focus on specific elements or show a broader view."),
        ],
    ),
    (
        "Special Effects", [
            ("Add Special Effects", "Introduce effects like glow, motion blur, or lens flare."),
            ("Remove Special Effects", "Eliminate existing special effects from the image."),
            ("Add Particles", "Insert particles like dust."),
            ("Remove Particles", "Remove existing particles from the image."),
        ],
    )
]


SEQUENCE_TEMPLATE = """Given an input image, write a sequence of {num} editing instructions for a instruction-based image editing model.
Each instruction should be simple, concise and belong to one of the valid atomic operations so that the full sequence can represent a complicated editing operation.
Don't include the purpose for the operation but only describe it in the instruction.

You need to pay attention to two critical issues.
1. As these instructions will be performed step-by-step, with each step's output image being the next step's input image, you need to consider the consistency for each instruction. \
For example, after removing an object, it should not be removed again. And after replacing one object with another, you can't alter the original object's color or other attributes.

2. Take consideration that this image editing model takes in the result of only one previous operation as input, therefore you need to prevent necessary information to be lost at each step. \
For example, "Replace an Object" or "Move an Object" should not be breakdown into "Remove an Object" and "Add an Object" as the information about the object would be lost for the addition. \
The same principle may apply to other operations.

Here are the valid options for atomic operations:
{options}

Explain your reasoning before give the answer.
"""


COMPOUND_TEMPLATE = """You are given an input image and a sequence of atomic editing instructions for an instruction-based image editing model.
Althought each instruction is atomic and simple, the full sequence can represent a complicated editing operation.

You need to write a single compound instruction that is equivalent to performing the editing sequence step-by-step.

Keep the language concise and technical. Don't include the purpose for the operation or any unnecessary information but only describe it in the instruction. \
For example, rather than "Remove the meshed fence from the image, introducing a sense of openness to the scene.", it is better to just write ""Remove the meshed fence".

Do not naively concatenate the atmoic operations together. Instead, write a natural, seamless instruction.
For example, rather than "Replace A with B, and change B's color to red", it is better to integrate these instructions as "Replace A with red B".

When composing this complicated instruction, you may change the order of atomic editing steps and break this instruction into multiple sentence as long as it is still equivalent to the sequentially performed atomic operations.

Explain your reasoning before give the answer.
"""


SIMPLIFY_TEMPLATE = """You are given an instruction for an instruction-based image editing model.
You are to refine the instruction to make it more concise and technical. Remove all the unnecessary information such as the purpose of the operation. \
For example, rather than "Remove the meshed fence from the image, introducing a sense of openness to the scene.", it is better to just write ""Remove the meshed fence".

Do not change the operation itself but only the expression of it.

Determine whether the instruction is already concise or needs simplification.
If it is concise, you can just copy it as is. Otherwise, provide a more concise version of the instruction.
"""

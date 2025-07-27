
ALIGNMENT_PROMPT_WO_RUBRIC = """
You are required to evaluate the result of an instruction-based image editing model.
Given an input image, an output image and a text instruction, you are required to access the output image \
based on whether the changes made to the input image align with the text instruction.

You are required to give two integer scores in [0, 10] based on the following criteria:
1. Instruction Following: whether the required changes occur in the output image, regardless of whether unnecessary changes are also made. \
10 means that all the changes required by the instruction occur in the output image, 0 means that no changes required by the instruction occur in the output image.
2. Identity Preservation: whether elements that should not be changed stay the same in the output image, regardless of whether required changes occur. \
10 means that no unnecessary changes occur in the output image, 0 means that all elements in the input image that should be kept the same are changed in the output image.

Note that these two scores should be graded independently, and a low score for one criterion should not affect the score for the other criterion.
For example, an output image that is identical to the input image should have an Instruction Following score of 0, but an Identity Preservation score of 10. \
Also, an output image that has no relevance with the input image should have an Identity Preservation score of 0 unless the instruction specifically orders the model to create a whole different image, \
but it should not affect the Instruction Following score as long as changes required by the instruction occur in the output.

If the instruction contains several atomic operations, evaluate the Instruction Following for each atomic operation separately and then average the scores as the assessment for Instruction Following.
"""


ALIGNMENT_PROMPT_W_RUBRIC = """
You are required to evaluate the result of an instruction-based image editing model.
Given an input image, an output image and a text instruction, you are required to access the output image \
based on whether the changes made to the input image align with the text instruction.

You are required to give two integer scores in [0, 10] based on the following criteria:
1. Instruction Following: whether the required changes occur in the output image, regardless of whether unnecessary changes are also made. \
10 means that all the changes required by the instruction occur in the output image, 0 means that no changes required by the instruction occur in the output image.
2. Identity Preservation: whether elements that should not be changed stay the same in the output image, regardless of whether required changes occur. \
10 means that no unnecessary changes occur in the output image, 0 means that all elements in the input image that should be kept the same are changed in the output image.

Here is the detailed rubric for Instruction Following:
* 10 (Perfect Instruction Following): \
All the required changes occur in the output image.
* 9 (Near Perfect Instruction Following with negligible deviations): Almost all instructed changes are present but negligible deviations exist \
(e.g., a tiny color variation such as the cat in the image is now black but the ears are grey).
* 7-8 (Strong Instruction Following with minor deviations): Most required changes are applied accurately. Minor deviations exist but do not substantially alter the intended modification \
(e.g., a car is changed to blue as instructed, but the reflection on its surface still contains a red tint).
* 5-6 (Moderate Instruction Following with noticeable deviations): The output reflects an attempt to follow instructions but with moderate errors \
(e.g., adding a required element but with incorrect attributes like color or shape).
* 3-4 (Weak Instruction Following with major deviations): Most required modifications are missing, incorrect, or only vaguely implemented. Significant elements from the instruction are misrepresented \
(e.g., when instructed to add a hat, a small, barely visible accessory is added to the head, but it does not resemble a proper hat).
* 1-2 (Minimal Instruction Following with severe deviations): A vague attempt is made, but the required modifications are either incorrect or so minimal that they do not fulfill the instruction \
(e.g., the instruction asks to remove a person from the image, but they are still visible, just slightly blurred or faded instead of being properly erased.).
* 0 (Complete failed Instruction Following): The output image is entirely unrelated to the instruction.

Here is the detailed rubric for Identity Preservation:
* 10 (Perfect Identity Preservation): All key elements that should remain unchanged are completely preserved and indistinguishable from the input \
(e.g., a person's face, expression, and proportions remain completely unchanged except for the required edits).
* 9 (Near Perfect Identity Preservation with negligible distortion): Key elements that should remain unchanged are preserved with negligible distortion \
(e.g., A person's face is identical except for a tiny, imperceptible variation in hair texture).
* 7-8 (Strong Identity Preservation with minor distortion): Small details of the key elements may have changed, but they do not significantly disrupt the overall identity \
(e.g., a pet's fur pattern remains mostly accurate, but a minor detail like a stripe or spot is different).
* 5-6 (Moderate Identity Preservation with noticeable distortion): Most of the key elements remain recognizable but with noticeable distortions \
(e.g., the instruction asks to change a car's color, but the car's shape or size is modified along with the color).
* 3-4 (Weak Identity Preservation with major distortion): Key elements maintain a general resemblance but noticeable changes are present \
(e.g., the instruction asks to brighten the sky, but additional buildings in the background appear or disappear).
* 1-2 (Minimal Identity Preservation with severe distortion): Most key elements are significantly altered or replaced. The key elements in the output retain only minor aspects of the original, but major features are incorrect \
(e.g., a person's face is still a human face, but it no longer resembles the original person at all).
* 0 (Complete failed Identity Preservation): All key elements that should remain unchanged are altered, distorted, or missing.

Note that these two scores should be graded independently, and a low score for one criterion should not affect the score for the other criterion.
For example, an output image that is identical to the input image should have an Instruction Following score of 0, but an Identity Preservation score of 10. \
Also, an output image that has no relevance with the input image should have an Identity Preservation score of 0 unless the instruction specifically orders the model to create a whole different image, \
but it should not affect the Instruction Following score as long as changes required by the instruction occur in the output.

If the instruction contains several atomic operations, evaluate the Instruction Following for each atomic operation separately and then average the scores as the assessment for Instruction Following.
"""


PROMPT_TEMPLATE = """The first image is the input image and the second image is the output image.
The text instruction is:
{instruction}

If the instruction contains several atomic operations, evaluate the Instruction Following for each atomic operation separately and then average the scores as the assessment for Instruction Following.
"""

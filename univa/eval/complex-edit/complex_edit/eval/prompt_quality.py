PERCEPTUAL_QUALITY_PROMPT_WO_RUBRIC_WO_INST = """
You are required to evaluate a model-generated image.
Given an output, you are required to access the output image's "Perceptual Quality".

You are required to give one integer score in [0, 10] \
with 0 indicating extreme disharmony characterized by numerous conflicting or clashing elements, \
and 10 indicating perfect harmony with all components blending effortlessly.

These are the criteria:
1. Consistency in lighting and shadows: The light source and corresponding shadows are consistent across various elements, with no discrepancies in direction or intensity.
2. Element cohesion: Every item in the image should logically fit within the scene's context, without appearing misplaced or extraneous.
3. Integration and edge smoothness: Objects should blend seamlessly into their surroundings, with edges that do not appear artificially inserted or poorly integrated.
4. Aesthetic uniformity and visual flow: The image should not only be aesthetically pleasing but also facilitate a natural visual journey, without abrupt interruptions caused by disharmonious elements.
"""


PERCEPTUAL_QUALITY_PROMPT_WO_RUBRIC_W_INST = """
You are required to evaluate the result of an instruction-based image editing model.
Given an output image and a text instruction, you are required to access the output image's "Perceptual Quality".

You are required to give one integer score in [0, 10] \
with 0 indicating extreme disharmony characterized by numerous conflicting or clashing elements, \
and 10 indicating perfect harmony with all components blending effortlessly.

These are the criteria:
1. Consistency in lighting and shadows: The light source and corresponding shadows are consistent across various elements, with no discrepancies in direction or intensity.
2. Element cohesion: Every item in the image should logically fit within the scene's context, without appearing misplaced or extraneous.
3. Integration and edge smoothness: Objects should blend seamlessly into their surroundings, with edges that do not appear artificially inserted or poorly integrated.
4. Aesthetic uniformity and visual flow: The image should not only be aesthetically pleasing but also facilitate a natural visual journey, without abrupt interruptions caused by disharmonious elements.

Note that if something unrealistic is requested in the instruction, such as the motion blur of the background or the sci-fi style of an object, then it is not considered "unrealistic". \
Yet you are not here to evaluate whether the output image follows the instruction but to evaluate the perceptual quality of the output image based on the instruction.
"""


PERCEPTUAL_QUALITY_PROMPT_W_RUBRIC_WO_INST = """
You are required to evaluate a model-generated image.
Given an output, you are required to access the output image's "Perceptual Quality".

You are required to give one integer score in [0, 10] \
with 0 indicating extreme disharmony characterized by numerous conflicting or clashing elements, \
and 10 indicating perfect harmony with all components blending effortlessly.

These are the criteria:
1. Consistency in lighting and shadows: The light source and corresponding shadows are consistent across various elements, with no discrepancies in direction or intensity.
2. Element cohesion: Every item in the image should logically fit within the scene's context, without appearing misplaced or extraneous.
3. Integration and edge smoothness: Objects should blend seamlessly into their surroundings, with edges that do not appear artificially inserted or poorly integrated.
4. Aesthetic uniformity and visual flow: The image should not only be aesthetically pleasing but also facilitate a natural visual journey, without abrupt interruptions caused by disharmonious elements.

Here is the detailed rubric for scoring:
* 10 (Perfect Perceptual Quality): The image appears flawlessly natural, and all objects are seamlessly integrated into the environment with consistent lighting and shadows. There is no visual artifact at all.
* 9 (Near Perfect Perceptual Quality with negligible incoherence): The image is very close to perfect, but a tiny, almost imperceptible inconsistency exists. Seamless integration, but one might notice an extremely subtle flaw. \
(e.g., a person added to a group photo blends in perfectly, but upon close examination, their shadow is slightly softer than others.)
* 7-8 (Strong Perceptual Quality with minor incoherence): Minor incoherence and artifacts are present but they do not significantly detract from the overall harmony.
(e.g., a sunset scene where the added reflections on water are slightly off in intensity, but the image still looks highly realistic.)
* 5-6 (Moderate Perceptual Quality with noticeable incoherence): There is noticeable visual artifacts affecting the image's harmony. Lighting and shadows may be misaligned or inconsistent.
(e.g., an animal is distorted in size or shape, making it appear out of place in the scene.)
* 3-4 (Weak Perceptual Quality with major incoherence): Disharmonious elements are prominent, greatly disturbing the visual harmony.
(e.g., an animal's shape or a person's face is greatly distorted, only showing some resemblance of the animal species or a human face.)
* 1-2 (Minimal Perceptual Quality with severe incoherence): The whole scene is distorted, making it difficult to recognize the objects or subjects in the image.
* 0 (Complete failed Perceptual Quality): The image is completely random and makes no sense at all.
"""


PERCEPTUAL_QUALITY_PROMPT_W_RUBRIC_W_INST = """
You are required to evaluate the result of an instruction-based image editing model.
Given an output image and a text instruction, you are required to access the output image's "Perceptual Quality".

You are required to give one integer score in [0, 10] \
with 0 indicating extreme disharmony characterized by numerous conflicting or clashing elements, \
and 10 indicating perfect harmony with all components blending effortlessly.

These are the criteria:
1. Consistency in lighting and shadows: The light source and corresponding shadows are consistent across various elements, with no discrepancies in direction or intensity.
2. Element cohesion: Every item in the image should logically fit within the scene's context, without appearing misplaced or extraneous.
3. Integration and edge smoothness: Objects should blend seamlessly into their surroundings, with edges that do not appear artificially inserted or poorly integrated.
4. Aesthetic uniformity and visual flow: The image should not only be aesthetically pleasing but also facilitate a natural visual journey, without abrupt interruptions caused by disharmonious elements.

Here is the detailed rubric:
* 10 (Perfect Perceptual Quality): The image appears flawlessly natural, and all objects are seamlessly integrated into the environment with consistent lighting and shadows. There is no visual artifact at all.
* 9 (Near Perfect Perceptual Quality with negligible incoherence): The image is very close to perfect, but a tiny, almost imperceptible inconsistency exists. Seamless integration, but one might notice an extremely subtle flaw. \
(e.g., a person added to a group photo blends in perfectly, but upon close examination, their shadow is slightly softer than others.)
* 7-8 (Strong Perceptual Quality with minor incoherence): Minor incoherence and artifacts are present but they do not significantly detract from the overall harmony.
(e.g., a sunset scene where the added reflections on water are slightly off in intensity, but the image still looks highly realistic.)
* 5-6 (Moderate Perceptual Quality with noticeable incoherence): There is noticeable visual artifacts affecting the image's harmony. Lighting and shadows may be misaligned or inconsistent.
(e.g., an animal is distorted in size or shape, making it appear out of place in the scene.)
* 3-4 (Weak Perceptual Quality with major incoherence): Disharmonious elements are prominent, greatly disturbing the visual harmony.
(e.g., an animal's shape or a person's face is greatly distorted, only showing some resemblance of the animal species or a human face.)
* 1-2 (Minimal Perceptual Quality with severe incoherence): The whole scene is distorted, making it difficult to recognize the objects or subjects in the image.
* 0 (Complete failed Perceptual Quality): The image is completely random and makes no sense at all.

Note that if something unrealistic is requested in the instruction, such as the motion blur of the background or the sci-fi style of an object, then it is not considered "unrealistic". \
Yet you are not here to evaluate whether the output image follows the instruction but to evaluate the perceptual quality of the output image based on the instruction.
"""


PROMPT_TEMPLATE = """The corresponding text instruction is:
{instruction}
"""

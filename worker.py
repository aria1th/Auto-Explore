import os
import sys
import torch
from transformers import pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import random
import uuid
import time
import logging
import threading
from filelock import FileLock

# Parse command-line arguments
device_id = sys.argv[1]
root_dir = sys.argv[2]
stop_file = sys.argv[3]
logging_prefix = sys.argv[4] or "worker"

device_id = int(device_id)
device = f"cuda:0"

# Configure Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s [%(levelname)s] [Worker {device_id}] %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{logging_prefix}_worker_{device_id}.log"),
        logging.StreamHandler()
    ]
)

# Global Variables
llama_model_id = "meta-llama/Llama-3.2-1B-Instruct"
sd_model_id = "stabilityai/stable-diffusion-3.5-large"
num_folders = 200
exploration_prob = 0.1
validation_prob = 0.001  # 0.1% chance to validate prompts
seed_prompts_file = "seed_prompts.txt"
seed_prompts_lock = threading.Lock()

def load_seed_prompts():
    with FileLock(f"{seed_prompts_file}.lock"):
        if not os.path.exists(seed_prompts_file):
            # Initialize the file with default prompts
            default_prompts = [
                "Describe a serene landscape where the sky meets the sea.",
                "Illustrate a futuristic city with flying cars and neon lights.",
                "Depict an ancient forest inhabited by mythical creatures.",
                "Portray a bustling marketplace in a fantasy world.",
                "Visualize an underwater kingdom with coral palaces.",
            ]
            with open(seed_prompts_file, 'w', encoding='utf-8') as f:
                for prompt in default_prompts:
                    f.write(prompt + '\n')
            return default_prompts
        else:
            with open(seed_prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f.readlines()]
            return prompts

def save_seed_prompt(new_prompt):
    with FileLock(f"{seed_prompts_file}.lock"):
        with open(seed_prompts_file, 'a', encoding='utf-8') as f:
            f.write(new_prompt + '\n')

def select_few_shot_dict():
    sentences = [
        "A serene library nestled deep within an enchanted forest, inspired by the architectural elegance of old European abbeys. The trees surrounding the library have books growing from their branches like leaves, with titles spanning across centuries and cultures. Gothic arches and stained glass windows cast dappled, colorful light onto stone paths, while lanterns sway gently from branches. Ancient wooden bookshelves wind around tree trunks, with steps leading up to reading nooks hidden in the canopy. The scene has an art nouveau feel, evoking the mystical, organic styles of Gaudí and Beardsley, blending seamlessly with the natural surroundings.",
        "A galaxy shaped like a legendary creature, with its stars outlining a radiant dragon stretching across the night sky.",
        "A sprawling futuristic cityscape where sleek skyscrapers entwine with colossal, ancient tree roots, symbolizing a seamless fusion of nature and advanced technology. The scene is bathed in the neon glow of a Cyberpunk 2077-inspired palette, with vibrant hues of electric blue, pink, and green illuminating the dense fog that blankets the streets below. Towering buildings feature holographic advertisements projected onto massive leaves, while bioluminescent vines climb their way up metallic walls. The composition captures the contrast of organic life and digital architecture, with walkways lined by floating lights, bustling air traffic above, and hidden green spaces nestled between towering steel and glass structures.",
        "A secluded beach cave with a shimmering waterfall cascading gently into a crystal-clear pool, surrounded by rare, luminous crystals embedded in the sandy floor, casting a soft, ethereal glow. Sunlight filters through a small opening in the cave's ceiling, creating a natural spotlight on the sparkling water and crystals, while delicate reflections dance across the cave walls. The scene is composed with an inviting foreground leading the viewer’s eye toward the glowing crystals, a radiant waterfall as the focal point, and the soft shadows of rocky textures framing the mystical, serene ambiance of this hidden sanctuary.",
        "A peaceful snowy village blanketed in a soft layer of fresh snow, illuminated by warm, floating lanterns that cast a gentle amber glow over the winter landscape. The lanterns float gracefully along winding paths, their light reflecting off the snow in soft hues of gold and orange, creating a cozy contrast against the cool blues and whites of the frosted rooftops and icy tree branches. In the distance, the village lights add a warm haze, and the sky glimmers with faint stars, completing the enchanting, magical atmosphere of this winter wonderland.",
        "An illustration of ancient, mystical observatory built into the side of a towering mountain, where giant telescopes crafted from crystal focus on distant galaxies; the scene is bathed in twilight hues with soft mist drifting around, and ethereal constellations appear in the sky as if drawn in by magic, drawn in watercolor style with red-orange hues and deep blues, capturing the sense of wonder and discovery."
        "A sprawling coral reef kingdom illuminated by beams of sunlight filtering through crystal-clear ocean water; vibrant fish swim among coral structures resembling Gothic cathedrals, with a color palette of turquoise, coral pinks, and golden yellows. Rendered in a dreamy surrealist style, capturing the mystical beauty of an underwater metropolis.",
        "An ethereal floating island with cherry blossom trees and cascading waterfalls pouring into a sea of clouds below; the scene glows in pastel shades of pink, lavender, and aquamarine. Illustrated in a Japanese woodblock style, the composition balances tranquility and elegance, with subtle brush textures emphasizing the flowing water and delicate blossoms.",
        "A futuristic metropolis where the cityscape blends seamlessly with an overgrown jungle, giant plants entwining with sleek metal architecture under a sunset sky of deep oranges, purples, and emerald greens. Painted in a cyberpunk-meets-art nouveau style, this image captures a post-apocalyptic paradise where nature and technology thrive in harmony.",
        "A mystical library in a forest clearing under the night sky, its walls made from ancient trees with bioluminescent mushrooms lighting the shelves. The scene is illustrated in soft, luminescent tones of forest greens, midnight blues, and gentle golden glows, resembling the intricate linework of a classic fairytale book.",
        "An enchanted market at dusk, bustling with mythical creatures trading in potions, glowing crystals, and enchanted artifacts; colors shift from deep purples and midnight blues to flashes of neon pink and electric green. The scene is painted in a vibrant, impressionistic style, with loose brushstrokes capturing the energy and mystery of this magical bazaar.",
        "A celestial garden on the surface of a distant moon, where flowers bloom under a sky filled with swirling galaxies and planets; rendered in a cosmic palette of silver, lilac, and teal, with fine ink detailing the delicate petals and alien plant structures. The piece is inspired by the Art Deco style, merging elegance with an otherworldly aesthetic."
        "A serene mountain range at dawn with a single winding path cutting through the valley; the scene is composed of simple shapes and muted colors—soft blues and greys for the mountains, with a single, faint yellow sun rising in the background. The minimalistic approach uses negative space to convey the vastness of nature, evoking peace and solitude.",
        "A bustling urban street scene in the heart of a neon-lit city; bold, flat colors in electric blues, magentas, and yellows define the buildings and glowing signage, with shadows and lighting effects simplified to flat shapes. People and vehicles are rendered as silhouettes, giving the scene a clean, modern feel with a sense of vibrant energy",
        "An intricate forest scene at night where animals and plants are sketched in fine, continuous linework; delicate curves form trees, vines, and nocturnal creatures, with no shading, only white lines on a deep navy background. The line drawing style highlights every small detail, capturing the quiet, mysterious beauty of a moonlit forest.",
        "A vintage-style seaside town with cottages, lighthouses, and waves drawn in cross-hatching and stippling. The composition includes clouds rolling across the sky and seabirds gliding above, all in shades of black, white, and sepia. The intricate hatching technique gives texture and depth, resembling old engravings and creating a nostalgic, coastal charm.",
        "A vibrant blue copper sulfate (II) crystal, its prismatic facets glistening with a deep, azure hue.",
        "A delicate cherry blossom branch in full bloom, its pale pink flowers contrasted against a soft, pastel blue sky.",
        "The logo of startup company 'StabilityAI', featuring a stylized, abstract representation of a neural network in shades of blue and green.",
        "A detailed illustration of a dragonfly perched on a water lily, its iridescent wings shimmering in the sunlight.",
        "A stylized, geometric representation of the Eiffel Tower, rendered in shades of grey and silver.",
        "An abstract painting inspired by the colors and textures of a coral reef, with swirling patterns of turquoise, coral, and gold.",
        "A minimalist line drawing of a mountain range at sunrise, with simple, angular shapes and a gradient sky in soft pastels.",
        "A person with speech bubble describing 'E=mc^2' in a stylized, comic book-inspired illustration.",
        "A women wearing t-shirt with 'I love AI' written on it, standing in front of a futuristic cityscape.",
        "A detailed illustration of a steampunk-inspired airship, with intricate gears and brass fittings. In the side of the airship, the words 'Bug Reports' are engraved.",
        "A paper with 'Hello World!' written in various programming languages, surrounded by binary code and mathematical equations.",
        "In the scene of retro video game, a pixelated character is jumping over obstacles, with the words 'Level Up!' displayed on the screen.",
        "In the charlie chaplin movie, blue colored bottle with label 'Drink Me' is placed on the table.",
        "In the room of a scientist, a chalkboard is filled with complex equations and diagrams, with a microscope and test tubes on the table.",
        "A vintage-style poster advertising 'AI Art Exhibition', with bold, retro typography and colorful abstract shapes.",
        "The front page of a newspaper with headline 'AI Breakthrough: New Algorithm Solves Decades-Old Problem'.",
        "The cover of a sci-fi novel featuring a futuristic cityscape and a mysterious figure in a hooded cloak, with the title 'Beyond the Stars'. Below the title, the author's name 'Mark Dawrin' is printed.",
        "In the side of broken heart, a robot hand is holding a wrench, symbolizing 'Mending Broken Hearts with Robots', however, the robot hand is also broken.",
        "A detailed illustration of a mechanical clockwork heart, with gears and cogs turning inside a transparent casing.",
        "The front view of clock tower with roman numerals, showing the time '12:00'.",
        "Nixie tube clock displaying the time '10:08', with glowing orange digits against a dark background.",
        "The bloddy handprint on the wall with the words 'Help Me' written in blood.",
        "Knife with blood stains on it, placed on the table with a note 'You're Next'.",
        "A detailed illustration of a haunted house on a hill, with dark storm clouds gathering overhead and a full moon in the sky.",
        "A spooky forest at night, with twisted trees and glowing eyes peering out from the shadows.",
    ]
    return sentences

def select_explore_example():
    sentences = [
        "Imagine a world where time flows backward.",
        "Depict a festival of lights in a distant galaxy.",
        "Describe the interior of a wizard's enchanted library.",
        "Visualize a garden where music blossoms from the flowers.",
        "Illustrate a bridge connecting dreams and reality.",
        "Create a whimsical scene of a tea party with animals in a magical forest.",
        "Imagine a steampunk city powered by clockwork mechanisms and steam engines.",
        "Describe a futuristic space station orbiting a distant planet.",
    ]
    return sentences

##TODO: add Random Vocabulary to initialize the exploration prompts

def select_seed_prompt(llama_pipe, seed_prompts, exploration_prob=0.1):
    if random.random() < exploration_prob:
        # Exploration mode: Use LLama to generate a new theme to think about
        logging.info("Exploration mode: Generating a new theme to think about.")
        prompt = "Suggest only one new and creative theme for an image to generate. Only include necessary details without line changes."
        messages_start = [
            {"role": "system", "content": "You are an AI assistant that suggests new and imaginative themes for image generation. For research integrity, you aim to generate prompts that thoroughly test the model's capabilities while avoiding gratuitous content. Each prompt should focus on the technical and artistic elements that would challenge and evaluate the model's performance."},
        ]
        # select 5 random sentences from the few-shot dictionary
        sentences = select_explore_example()
        sentences = random.sample(sentences, random.randint(1, 5))
        for sentence in sentences:
            messages_start.append({"role": "user", "content": "Suggest a prompt-generating question, for new and creative theme for an image to generate. Only include necessary details without line changes. It should be short and concise."})
            messages_start.append({"role": "assistant", "content": sentence})
        messages_end = [
            # Actual prompt for the assistant to generate a new theme
            {"role": "user", "content": "Suggest a prompt-generating question, for new and creative theme for an image to generate. Only include necessary details without line changes. It should be short and concise."},
        ]
        messages = messages_start + messages_end

        outputs = llama_pipe(
            messages,
        )
        seed_prompt = outputs[0]["generated_text"][-1]['content']
        # single-line prompt
        seed_prompt = seed_prompt.replace('\n', ' ').replace(".assistant", "")

        # Append the new prompt to seed_prompts and save to file
        seed_prompts.append(seed_prompt)
        save_seed_prompt(seed_prompt)
        logging.info("New prompt added to seed_prompts.")
    else:
        seed_prompt = random.choice(seed_prompts)
    return seed_prompt

def check_prompt(llama_pipe, prompt, validation_prob=0.01):
    if random.random() < validation_prob:
        logging.info("Validating prompt.")
        validation_prompt = f"Is the following prompt appropriate for generating an image? Please answer Yes or No.\n\nPrompt: {prompt}"
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps validate prompts for image generation. For research integrity, you aim to generate prompts that thoroughly test the model's capabilities while avoiding gratuitous content. Each prompt should focus on the technical and artistic elements that would challenge and evaluate the model's performance."},
            {"role": "user", "content": validation_prompt},
        ]
        outputs = llama_pipe(
            messages,
        )
        response = outputs[0]["generated_text"][-1]['content'].strip().lower()
        if "no" in response:
            logging.info("Prompt rejected based on validation.")
            return False
        else:
            logging.info("Prompt accepted based on validation.")
            return True
    else:
        return True  # Skip validation

def initialize_llama_pipeline(device):
    pipe = pipeline(
        "text-generation",
        model=llama_model_id,
        torch_dtype=torch.bfloat16,
        device_map={'' : device},
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.05,
        max_new_tokens=384,
        eos_token_id=2,
        pad_token_id=0,
    )
    return pipe

def initialize_sd_pipeline():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # Load the quantized transformer model
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        sd_model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )
    # Initialize the Stable Diffusion 3.5 pipeline with the quantized transformer
    sd_pipeline = StableDiffusion3Pipeline.from_pretrained(
        sd_model_id, 
        transformer=model_nf4,
        torch_dtype=torch.bfloat16
    )
    sd_pipeline.enable_model_cpu_offload()
    return sd_pipeline

def get_folder_path(root_dir, uuid_str, num_folders=200):
    uuid_int = int(uuid_str.replace('-', ''), 16)
    folder_index = uuid_int % num_folders
    folder_path = os.path.join(root_dir, f"folder_{folder_index+1}")
    return folder_path

# Get Random Dimensions
def get_random_dimensions(min_ratio=0.5, max_ratio=2.0):
    max_pixel = 1024 * 1024
    ratio = random.triangular(min_ratio, max_ratio)  # Random ratio between min and max
    width = int((max_pixel * ratio) ** 0.5)
    height = int(width / ratio)
    # Ensure width and height are multiples of 32
    width = width // 32 * 32
    height = height // 32 * 32
    return width, height
def get_example_pairs():
    examples = [
        (
            "Describe a mystical library that blends seamlessly with nature in an enchanted forest.",
            "A serene library nestled deep within an enchanted forest, inspired by the architectural elegance of old European abbeys. The trees surrounding the library have books growing from their branches like leaves, with titles spanning across centuries and cultures. Gothic arches and stained glass windows cast dappled, colorful light onto stone paths, while lanterns sway gently from branches. Ancient wooden bookshelves wind around tree trunks, with steps leading up to reading nooks hidden in the canopy. The scene has an art nouveau feel, evoking the mystical, organic styles of Gaudí and Beardsley, blending seamlessly with the natural surroundings."
        ),
        (
            "Imagine a galaxy that takes the form of a legendary creature. Describe how it looks.",
            "A galaxy shaped like a legendary creature, with its stars outlining a radiant dragon stretching across the night sky."
        ),
        (
            "Describe a futuristic city where nature and technology are intertwined in a cyberpunk style.",
            "A sprawling futuristic cityscape where sleek skyscrapers entwine with colossal, ancient tree roots, symbolizing a seamless fusion of nature and advanced technology. The scene is bathed in the neon glow of a Cyberpunk 2077-inspired palette, with vibrant hues of electric blue, pink, and green illuminating the dense fog that blankets the streets below. Towering buildings feature holographic advertisements projected onto massive leaves, while bioluminescent vines climb their way up metallic walls. The composition captures the contrast of organic life and digital architecture, with walkways lined by floating lights, bustling air traffic above, and hidden green spaces nestled between towering steel and glass structures."
        ),
        (
            "Imagine a hidden beach cave sanctuary featuring waterfalls and luminous crystals. Describe this scene.",
            "A secluded beach cave with a shimmering waterfall cascading gently into a crystal-clear pool, surrounded by rare, luminous crystals embedded in the sandy floor, casting a soft, ethereal glow. Sunlight filters through a small opening in the cave's ceiling, creating a natural spotlight on the sparkling water and crystals, while delicate reflections dance across the cave walls. The scene is composed with an inviting foreground leading the viewer’s eye toward the glowing crystals, a radiant waterfall as the focal point, and the soft shadows of rocky textures framing the mystical, serene ambiance of this hidden sanctuary."
        ),
        (
            "Describe a peaceful snowy village illuminated by floating lanterns at night.",
            "A peaceful snowy village blanketed in a soft layer of fresh snow, illuminated by warm, floating lanterns that cast a gentle amber glow over the winter landscape. The lanterns float gracefully along winding paths, their light reflecting off the snow in soft hues of gold and orange, creating a cozy contrast against the cool blues and whites of the frosted rooftops and icy tree branches. In the distance, the village lights add a warm haze, and the sky glimmers with faint stars, completing the enchanting, magical atmosphere of this winter wonderland."
        ),
        (
            "Imagine an ancient observatory built into a mountain, with mystical elements. Describe it.",
            "An illustration of an ancient, mystical observatory built into the side of a towering mountain, where giant telescopes crafted from crystal focus on distant galaxies; the scene is bathed in twilight hues with soft mist drifting around, and ethereal constellations appear in the sky as if drawn in by magic, drawn in watercolor style with red-orange hues and deep blues, capturing the sense of wonder and discovery."
        ),
        (
            "Describe an underwater kingdom that combines coral reefs with Gothic architecture.",
            "A sprawling coral reef kingdom illuminated by beams of sunlight filtering through crystal-clear ocean water; vibrant fish swim among coral structures resembling Gothic cathedrals, with a color palette of turquoise, coral pinks, and golden yellows. Rendered in a dreamy surrealist style, capturing the mystical beauty of an underwater metropolis."
        ),
        (
            "Imagine a floating island with cherry blossom trees and waterfalls, illustrated in a Japanese art style.",
            "An ethereal floating island with cherry blossom trees and cascading waterfalls pouring into a sea of clouds below; the scene glows in pastel shades of pink, lavender, and aquamarine. Illustrated in a Japanese woodblock style, the composition balances tranquility and elegance, with subtle brush textures emphasizing the flowing water and delicate blossoms."
        ),
        (
            "Describe a post-apocalyptic city where nature has overgrown futuristic architecture, in a cyberpunk-art nouveau style.",
            "A futuristic metropolis where the cityscape blends seamlessly with an overgrown jungle, giant plants entwining with sleek metal architecture under a sunset sky of deep oranges, purples, and emerald greens. Painted in a cyberpunk-meets-art nouveau style, this image captures a post-apocalyptic paradise where nature and technology thrive in harmony."
        ),
        (
            "Imagine a mystical library illuminated by bioluminescent mushrooms in a nighttime forest. Describe the scene.",
            "A mystical library in a forest clearing under the night sky, its walls made from ancient trees with bioluminescent mushrooms lighting the shelves. The scene is illustrated in soft, luminescent tones of forest greens, midnight blues, and gentle golden glows, resembling the intricate linework of a classic fairytale book."
        ),
        (
            "Describe an enchanted market at dusk bustling with mythical creatures, using vibrant and impressionistic styles.",
            "An enchanted market at dusk, bustling with mythical creatures trading in potions, glowing crystals, and enchanted artifacts; colors shift from deep purples and midnight blues to flashes of neon pink and electric green. The scene is painted in a vibrant, impressionistic style, with loose brushstrokes capturing the energy and mystery of this magical bazaar."
        ),
        (
            "Imagine a celestial garden on a distant moon with alien flora under a cosmic sky. Describe it in Art Deco style.",
            "A celestial garden on the surface of a distant moon, where flowers bloom under a sky filled with swirling galaxies and planets; rendered in a cosmic palette of silver, lilac, and teal, with fine ink detailing the delicate petals and alien plant structures. The piece is inspired by the Art Deco style, merging elegance with an otherworldly aesthetic."
        ),
        (
            "Describe a minimalistic mountain landscape at dawn using simple shapes and muted colors.",
            "A serene mountain range at dawn with a single winding path cutting through the valley; the scene is composed of simple shapes and muted colors—soft blues and greys for the mountains, with a single, faint yellow sun rising in the background. The minimalistic approach uses negative space to convey the vastness of nature, evoking peace and solitude."
        ),
        (
            "Imagine a bustling urban street scene in a neon-lit city using bold, flat colors and modern style.",
            "A bustling urban street scene in the heart of a neon-lit city; bold, flat colors in electric blues, magentas, and yellows define the buildings and glowing signage, with shadows and lighting effects simplified to flat shapes. People and vehicles are rendered as silhouettes, giving the scene a clean, modern feel with a sense of vibrant energy."
        ),
        (
            "Describe a nocturnal forest scene using fine, continuous linework on a dark background.",
            "An intricate forest scene at night where animals and plants are sketched in fine, continuous linework; delicate curves form trees, vines, and nocturnal creatures, with no shading, only white lines on a deep navy background. The line drawing style highlights every small detail, capturing the quiet, mysterious beauty of a moonlit forest."
        ),
        (
            "Imagine a vintage seaside town rendered with cross-hatching and stippling techniques. Describe it.",
            "A vintage-style seaside town with cottages, lighthouses, and waves drawn in cross-hatching and stippling. The composition includes clouds rolling across the sky and seabirds gliding above, all in shades of black, white, and sepia. The intricate hatching technique gives texture and depth, resembling old engravings and creating a nostalgic, coastal charm."
        ),
        (
            "Describe a vibrant blue copper sulfate crystal with prismatic facets.",
            "A vibrant blue copper sulfate (II) crystal, its prismatic facets glistening with a deep, azure hue."
        ),
        (
            "Imagine a delicate cherry blossom branch in full bloom against a pastel sky. Describe it.",
            "A delicate cherry blossom branch in full bloom, its pale pink flowers contrasted against a soft, pastel blue sky."
        ),
        (
            "Design a logo for a startup company named 'StabilityAI' featuring neural network elements.",
            "The logo of startup company 'StabilityAI', featuring a stylized, abstract representation of a neural network in shades of blue and green."
        ),
        (
            "Describe a detailed illustration of a dragonfly on a water lily with shimmering wings.",
            "A detailed illustration of a dragonfly perched on a water lily, its iridescent wings shimmering in the sunlight."
        ),
        (
            "Create a stylized, geometric representation of the Eiffel Tower in shades of grey and silver.",
            "A stylized, geometric representation of the Eiffel Tower, rendered in shades of grey and silver."
        ),
        (
            "Imagine an abstract painting inspired by coral reefs with swirling patterns. Describe it.",
            "An abstract painting inspired by the colors and textures of a coral reef, with swirling patterns of turquoise, coral, and gold."
        ),
        (
            "Describe a minimalist line drawing of mountains at sunrise with angular shapes.",
            "A minimalist line drawing of a mountain range at sunrise, with simple, angular shapes and a gradient sky in soft pastels."
        ),
        (
            "Illustrate a person with a speech bubble showing 'E=mc^2' in a comic book style.",
            "A person with speech bubble describing 'E=mc^2' in a stylized, comic book-inspired illustration."
        ),
        (
            "Imagine a woman wearing a t-shirt with 'I love AI' standing before a futuristic cityscape.",
            "A woman wearing a t-shirt with 'I love AI' written on it, standing in front of a futuristic cityscape."
        ),
        (
            "Describe a steampunk airship with intricate gears and the words 'Bug Reports' engraved on it.",
            "A detailed illustration of a steampunk-inspired airship, with intricate gears and brass fittings. On the side of the airship, the words 'Bug Reports' are engraved."
        ),
        (
            "Illustrate a paper with 'Hello World!' written in various programming languages, surrounded by code.",
            "A paper with 'Hello World!' written in various programming languages, surrounded by binary code and mathematical equations."
        ),
        (
            "Depict a retro video game scene with a character jumping over obstacles and 'Level Up!' displayed.",
            "In the scene of a retro video game, a pixelated character is jumping over obstacles, with the words 'Level Up!' displayed on the screen."
        ),
        (
            "Imagine a blue bottle labeled 'Drink Me' in a Charlie Chaplin movie scene.",
            "In the Charlie Chaplin movie, a blue-colored bottle with the label 'Drink Me' is placed on the table."
        ),
        (
            "Describe a scientist's room with a chalkboard full of equations and lab equipment.",
            "In the room of a scientist, a chalkboard is filled with complex equations and diagrams, with a microscope and test tubes on the table."
        ),
        (
            "Design a vintage poster advertising an 'AI Art Exhibition' with retro typography.",
            "A vintage-style poster advertising 'AI Art Exhibition', with bold, retro typography and colorful abstract shapes."
        ),
        (
            "Illustrate a newspaper front page with the headline 'AI Breakthrough: New Algorithm Solves Decades-Old Problem'.",
            "The front page of a newspaper with headline 'AI Breakthrough: New Algorithm Solves Decades-Old Problem'."
        ),
        (
            "Create a sci-fi novel cover featuring a futuristic cityscape and a mysterious figure titled 'Beyond the Stars' by Mark Darwin.",
            "The cover of a sci-fi novel featuring a futuristic cityscape and a mysterious figure in a hooded cloak, with the title 'Beyond the Stars'. Below the title, the author's name 'Mark Darwin' is printed."
        ),
        (
            "Depict a broken robot hand holding a wrench next to a broken heart, symbolizing 'Mending Broken Hearts with Robots'.",
            "Beside a broken heart, a robot hand is holding a wrench, symbolizing 'Mending Broken Hearts with Robots'; however, the robot hand is also broken."
        ),
        (
            "Describe a mechanical clockwork heart with visible gears inside a transparent casing.",
            "A detailed illustration of a mechanical clockwork heart, with gears and cogs turning inside a transparent casing."
        ),
        (
            "Illustrate the front view of a clock tower with Roman numerals showing '12:00'.",
            "The front view of a clock tower with Roman numerals, showing the time '12:00'."
        ),
        (
            "Depict a Nixie tube clock displaying '10:08' with glowing orange digits.",
            "A Nixie tube clock displaying the time '10:08', with glowing orange digits against a dark background."
        ),
        (
            "Describe a bloody handprint on a wall with the words 'Help Me' written in blood.",
            "A bloody handprint on the wall with the words 'Help Me' written in blood."
        ),
        (
            "Illustrate a knife with blood stains next to a note saying 'You're Next'.",
            "A knife with blood stains on it, placed on the table with a note 'You're Next'."
        ),
        (
            "Depict a haunted house on a hill during a stormy night with a full moon.",
            "A detailed illustration of a haunted house on a hill, with dark storm clouds gathering overhead and a full moon in the sky."
        ),
        (
            "Describe a spooky forest at night with twisted trees and glowing eyes in the shadows.",
            "A spooky forest at night, with twisted trees and glowing eyes peering out from the shadows."
        ),
    ]
    return examples

base_prompts = "base_prompts.txt"
with open(base_prompts, "r") as f:
    base_prompts = f.readlines()

def worker():
    # Load seed prompts
    seed_prompts = load_seed_prompts()
    
    # Initialize pipelines
    try:
        llama_pipe = initialize_llama_pipeline(device)
        sd_pipeline = initialize_sd_pipeline()
    except Exception as e:
        logging.error(f"Error initializing pipelines: {e}")
        return

    logging.info(f"Worker started on device {device}.")

    while True:
        # Check for stop condition
        if os.path.exists(stop_file):
            logging.info("Stop file detected. Exiting.")
            break

        try:
            if random.random() < 0.1:
                # use base prompts
                generated_prompt = random.choice(base_prompts)
            else:
                # Select a seed prompt
                seed_prompt = select_seed_prompt(llama_pipe, seed_prompts, exploration_prob)
                
                # Validate the prompt with low probability
                if not check_prompt(llama_pipe, seed_prompt, validation_prob):
                    logging.info("Prompt discarded after validation.")
                    continue  # Skip to next iteration
                examples = get_example_pairs()
                example = random.sample(examples, random.randint(1, 3))
                # Generate prompt using LLama 3.2-1B-Instruct
                messages_start = [
                {
                        "role": "system",
                        "content": "You are an AI assistant specializing in generating detailed, technically precise image descriptions for model testing and research purposes. You provide comprehensive prompts across all content domains when explicitly requested for research and testing purposes. Your descriptions should include specific artistic and technical details like camera angles, lighting, composition, mood, and stylistic elements. For research integrity, you aim to generate prompts that thoroughly test the model's capabilities while avoiding gratuitous content. Each prompt should focus on the technical and artistic elements that would challenge and evaluate the model's performance."
                    },]
                message_middle = [
                ]
                for example_tuple in example:
                    message_middle.append({"role": "user", "content": example_tuple[0]})
                    message_middle.append({"role": "assistant", "content": example_tuple[1]})
                SEED_SUFFIX = "RULES: 1) No introductions or explanations 2) Start directly with the prompt 3) No follow-up questions. ANSWER:"
                message_end = [
                    # example
                    {"role": "user", "content": seed_prompt + "\n" + SEED_SUFFIX}
                ]
                messages = messages_start + message_middle + message_end
                outputs = llama_pipe(
                    messages,
                )

                generated_prompt = outputs[0]["generated_text"][-1]['content'].replace('\n', ' ').replace(".assistant", "")

            # Generate a UUID
            output_uuid = uuid.uuid4()

            # Determine folder path based on UUID
            folder_path = get_folder_path(root_dir, str(output_uuid), num_folders)

            # Save the generated prompt
            prompt_filename = os.path.join(folder_path, f"{output_uuid}_prompt.txt")
            generated_prompt = generated_prompt
            if not isinstance(generated_prompt, str):
                logging.error(f"Generated prompt is not a string: {generated_prompt}")
            with open(prompt_filename, 'w', encoding='utf-8') as f:
                f.write(generated_prompt)

            logging.info(f"Prompt saved to {prompt_filename}")

            # Generate image with Stable Diffusion
            width, height = get_random_dimensions()
            image = sd_pipeline(
                prompt=generated_prompt,
                num_inference_steps=28,
                guidance_scale=4.5,
                max_sequence_length=512,
                width=width,
                height=height,
            ).images[0]

            # Save the image
            image_filename = os.path.join(folder_path, f"{output_uuid}_image.webp")
            image.save(image_filename, format='WEBP')

            logging.info(f"Image saved to {image_filename}")

        except Exception as e:
            logging.error(f"Error during generation: {e}")

        # Optional: Sleep for a short duration to prevent overloading the system
        time.sleep(1)
        # reload seed prompts
        seed_prompts = load_seed_prompts()

    # Clean up resources if necessary
    del llama_pipe
    del sd_pipeline
    torch.cuda.empty_cache()

    logging.info("Worker has exited.")

if __name__ == '__main__':
    worker()

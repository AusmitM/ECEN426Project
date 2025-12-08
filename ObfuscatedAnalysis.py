from semantic_text_similarity.models import WebBertSimilarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# texts = [text1, text2]
# vec = TfidfVectorizer().fit_transform(texts)  # sparse matrix
# sim_matrix = cosine_similarity(vec[0], vec[1])
# print("TF-IDF cosine:", float(sim_matrix))


# This file uses the Obfuscated code from the 100 contest winner projects after the 2011 IOCCC competition and instruct the LLM to preform deobfuscation.
# The original obfuscated code is given to the LLM without any modification

# import re
# from openai import OpenAI
# from dotenv import load_dotenv
# import subprocess
# import tempfile
# import os

# from Metrics import CosineSimilarity, WebBertSim, GPT4AllSim
# from ModelCalls import GPT35TurboAnalysis
from Metrics import CosineSimilarity, WebBertSim
import matplotlib.pyplot as plt
import numpy as np
# DE_FOLDER_PATH = "./output/default"
# DCI_FOLDER_PATH = "./output/deadcode"
# CFF_FOLDER_PATH = "./output/cff"
# SS_FOLDER_PATH = "./output/split_strings"
# WO_FOLDER_PATH = "./output/wobfuscator"
# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

explanations = [
    # 0 - crypto.js
    """This file implements an arbitrary-precision integer (big number) library and related
cryptographic routines. It defines a BigInteger type with support for operations like
addition, subtraction, multiplication, division, modular reduction, exponentiation, and
conversion to and from strings or byte arrays. It also includes support code for modular
arithmetic (e.g., Classic and Montgomery reducers) and exponentiation routines that are
suitable for RSA-style public-key cryptography. The code is structured as a performance
benchmark that stresses big-integer arithmetic rather than providing any UI.""",

    # 1 - deltablue.js
    """This file is a JavaScript implementation of the DeltaBlue constraint-solving algorithm.
It defines an object model for constraints, variables, and strengths, and a planner that
incrementally maintains relationships among variables. Constraints can be unary or binary
and have different strengths that affect which relationships are enforced when conflicts
arise. The benchmark builds a small constraint network (for example, linked variables with
equality and stay constraints) and then runs the planner to adjust variable values and
propagate changes, exercising object allocation, method calls, and graph-based logic.""",

    # 2 - localStorage.js
    """This file implements a simple demo of using localStorage to remember a chosen background
color. It first shows a basic example of storing a username, then queries DOM elements for
a button container, an area whose background is controlled by CSS classes, and the trigger
button that inserts color buttons. On page load, it reads a 'colorFondo' value from
localStorage and applies either a default 'bg-dark' class or the stored class to the
background element. When the user clicks one of the dynamically inserted Bootstrap-style
color buttons, an event-delegation handler updates the background element’s class to match
the chosen color and writes the chosen class name back into localStorage so the preference
persists across reloads.""",

    # 3 - passwordGenerator.js
    """This file implements a configurable password generator wrapped in an immediately invoked
function. It reads settings from a form: the desired password length and toggles for
including symbols, numbers, uppercase letters, and lowercase letters. A configuration
object tracks which character categories are enabled, and a lookup object provides the
actual character sets. UI buttons increment or decrement the length and toggle categories,
updating CSS classes to reflect the state. When the user clicks the generate button, the
code builds a combined character array from all enabled categories and generates a random
password of the configured length by sampling from that array. The new password is written
into an input field, and clicking that field selects the text, copies it to the clipboard,
and briefly shows a “copied” notification. A password is also generated once at startup.""",

    # 4 - raytrace.js
    """This file contains a full ray tracer implemented in JavaScript. It defines a namespace
with classes for colors, vectors, rays, materials, shapes (such as spheres and planes),
lights, a camera, a scene, and a rendering engine. The engine casts rays from the camera
through each pixel of an image, computes intersections with the scene’s objects, and then
uses material and lighting calculations (including reflections and shading) to determine
the final color for each pixel. The code includes minimal pieces of a class framework to
support its object model and accumulates a numeric check value to verify correct rendering.
Its primary purpose is to serve as a computational benchmark for floating-point math and
object-heavy workloads rather than user interface logic.""",

    # 5 - richards.js
    """This file implements the Richards benchmark, which simulates an operating-system-like
task dispatcher. It defines a Scheduler that manages a fixed set of task control blocks
and Packet queues, and several task types (idle, worker, handler, and device tasks) that
communicate by sending and receiving packets. The runRichards function sets up a specific
network of tasks with different priorities and initial queues, then repeatedly schedules
tasks until the system quiesces. Along the way it tracks how many packets were queued and
how many tasks were put on hold, and at the end it checks those counts against expected
constants to confirm correct behavior. The benchmark stresses control flow, priority-based
scheduling, and message passing between tasks.""",

    # 6 - rockPaperScissors.js
    """This file implements a browser-based rock–paper–scissors game. It defines a game()
function that encapsulates all state, including player and computer scores. A start screen
is shown first, and clicking the play button fades in the match view. For each round,
click handlers on the rock, paper, and scissors buttons randomly choose the computer’s
move, animate both hand images with a shaking effect, and after a short delay compare the
player and computer choices. The code updates a winner text element to indicate whether
the player, computer, or neither won, and increments and redisplays the running scores.
The script calls game() at the end so the event handlers and UI transitions are initialized
automatically when the file is loaded.""",

    # 7 - toDoList.js
    """This file implements a minimal to-do list application. It grabs references to a form,
a text input, and a list container, then intercepts the form’s submit event to prevent a
page reload. If the input is empty, it calls a small helper that logs an 'empty' message;
otherwise it calls addItem, which creates a new list item element containing the entered
text and an inline delete button that calls removeItem(this). The new list item is inserted
at the top of the list, and the input is cleared and refocused for the next entry.
removeItem removes the corresponding list item element from the DOM, effectively deleting
that to-do entry. Overall, it provides basic add and remove functionality for a text-based
to-do list.""",

    # 8 - weather.js
    """This file implements a geolocation-based weather widget for the browser. Inside an
IIFE, it queries DOM elements for displaying the location name, temperature, unit label,
and description. If geolocation is available, it obtains the user’s latitude and longitude
and uses them to build a Dark Sky API URL via a CORS proxy. It fetches the current weather
JSON, extracts the temperature, a textual summary, and an icon code, and updates the DOM
accordingly. It also calls a helper function that configures a Skycons canvas icon based
on the reported icon name. A click handler on the temperature section toggles the display
between Fahrenheit and Celsius using the standard conversion formula, updating both the
numeric value and the unit label. All logic is wrapped so it runs automatically on page
load without exposing globals beyond what the HTML uses."""
]


GPT5DefaultExplanations = [
    """
    file1.js – Implements a complete big-integer arithmetic library and an RSA cryptosystem.
    It defines operations such as modular exponentiation, primality testing, and key handling,
    then builds an RSAKey class that supports encryption/decryption with PKCS#1 padding.
    At the bottom it includes hard-coded RSA key parameters and runs a self-test that encrypts
    and decrypts a sample message to ensure the implementation works correctly.
    """,

    """
    file2.js – A constraint-solving system patterned after the classic DeltaBlue benchmark.
    It defines constraints, variables, strengths, and a Planner that creates and executes
    dependency plans. The system propagates changes through a constraint graph and resolves
    constraints according to priority, functioning as a benchmark for incremental constraint
    solvers.
    """,

    """
    file3.js – A small UI script that stores a username and a chosen background color in
    localStorage. It dynamically generates Bootstrap-styled color buttons which, when clicked,
    update the page background and persist the selection across page reloads. Defaults to
    bg-dark if no saved color exists.
    """,

    """
    file4.js – A configurable password generator. A form allows users to select length and
    character categories (symbols, numbers, uppercase, lowercase). It builds a character pool,
    generates a random password into an input field, and supports copying via execCommand.
    The UI includes toggles, plus/minus controls, and a temporary "copied" alert.
    """,
    """
    file5.js – A Color class used in a ray-tracing context. It stores red/green/blue values
    and supports arithmetic (add, subtract, multiply), blending, brightness calculation,
    distance measurement, clamping, and scalar operations. It is a utility module enabling
    color computations for rendering.
    """,
    """
    file6.js – Implements the Richards benchmark, a synthetic workload for task scheduling.
    It defines scheduler structures, packet queues, and task types (Idle, Worker, Handler,
    Device). The benchmark runs tasks until quiescence and verifies expected counts, serving
    as a CPU/algorithm performance test.
    """,
    """
    file7.js – A browser rock-paper-scissors game. It manages UI transitions, button handlers,
    and animated hand images. When a player selects an option, the computer randomly chooses
    one, scores are updated, animations play, and the result ("Player Wins" etc.) is shown.
    """,

    """
    file8.js – A minimal todo-list implementation. It reads user input from a form, appends
    a new <li> containing the text and a remove button, and supports item deletion via a small
    'x' button. Entirely client-side and uses simple DOM manipulation.
    """,

    """
    file9.js – A weather widget using the browser's geolocation API and the Dark Sky service.
    After obtaining latitude/longitude, it fetches current weather, displays temperature,
    summary, and timezone, and uses Skycons to render an animated icon. Includes a °F/°C
    toggle on click.
    """
]
GPT4DefaultExplanations = [
    """"
file1.js - Arbitrary-Precision Integer Arithmetic
This file implements a custom BigInteger class that provides support for arbitrary-precision integer operations. It includes logic for arithmetic operations such as addition, multiplication, and conversion from strings. The code is heavily obfuscated, but its structure suggests usage in cryptographic systems or numerical applications where high-precision math is needed, such as RSA or blockchain technology.
""",

    """"
file2.js - Constraint-Based Layout System
This script implements a constraint-solving framework resembling the Cassowary algorithm. It includes classes like OrderedCollection and Strength to manage and prioritize constraints, which can be incrementally added, removed, or evaluated. The goal is likely layout management or planning systems, where relationships between variables need to be maintained through rules of strength or preference.
""",

    """"
file3.js - DOM Manipulation & Theme Selector
This file handles UI theme changes through button interactions and stores the chosen background color in localStorage for session persistence. It dynamically renders styled buttons (e.g., primary, warning, danger) and updates DOM classes accordingly. It also logs the stored username, indicating this could be part of a personalization demo or a learning exercise for localStorage and class management.
""",

    """"
file4.js - Password Generator Interface
This script allows users to generate secure passwords based on selected criteria like uppercase, lowercase, numbers, and symbols. It handles DOM interactions, updates the display with the generated password, and supports clipboard copying. The script is a common utility in security-conscious applications or teaching interfaces for JavaScript forms and user interaction.
""",

    """"
file5.js - Ray Tracer Color Class
This file defines a Color class under the Flog.RayTracer namespace, supporting operations like addition, scalar multiplication, clamping, and blending of RGB color components. These operations are essential in ray tracing or graphics simulations for manipulating light and surface properties. The logic is suitable for inclusion in a rendering engine or educational tool on color mathematics.
""",

    """"
file6.js - Benchmarking with Richards Scheduler
This file implements the Richards benchmark, which simulates a scheduler with multiple task types (Idle, Worker, Handler, Device). It manages task control blocks and packet queues to simulate multitasking and is used to test the performance of JavaScript engines or other runtimes. It serves as a performance benchmark for measuring task switching and computational throughput.
""",

    """"
file7.js - Rock Paper Scissors Game
A browser-based Rock-Paper-Scissors game that uses buttons for user interaction, randomly generates the computer’s move, and updates the score and visuals accordingly. The game logic includes animations and DOM manipulation. It is a typical example used in teaching event handling, arrays, and basic control flow in interactive web development.
""",

    """"
file8.js - To-Do List App
This file creates a basic to-do list interface using an HTML form. Users can submit tasks, which are added to the DOM as list items, each with a delete button. It checks for empty inputs and logs a warning message. This script is commonly used in beginner tutorials on JavaScript and front-end development for demonstrating event handling and dynamic content insertion.
""",

    """"
file9.js - Weather API Integration with Geolocation
This script fetches and displays weather data based on the user's geographic coordinates using the browser’s Geolocation API. It calls the Dark Sky API to retrieve temperature, summary, and icon data, then updates the DOM. It also includes a toggle to convert between Fahrenheit and Celsius. This example demonstrates asynchronous data fetching, API integration, and geolocation usage.
"""
]

GPT5_default_cosine_vals = []
GPT4_default_cosine_vals = []
GPT5_default_bert_vals = []
GPT4_default_bert_vals = []
GPT5_default_gpt5_vals = [1, 1, 1, 1, 0, 1, 1, 1, 1]
GPT4_default_gpt5_vals = [1, 1, 1, 1, 0, 1, 1, 1, 1]
for i in range(9):
    GPT5_default_cosine_vals.append(CosineSimilarity(explanations[i], GPT5DefaultExplanations[i]))
    GPT4_default_cosine_vals.append(CosineSimilarity(explanations[i], GPT4DefaultExplanations[i]))
    GPT5_default_bert_vals.append(WebBertSim(explanations[i], GPT5DefaultExplanations[i]))
    GPT4_default_bert_vals.append(WebBertSim(explanations[i], GPT4DefaultExplanations[i]))

print("GPT-5 Default Cosine Similarities:", GPT5_default_cosine_vals)
print("GPT-4 Default Cosine Similarities:", GPT4_default_cosine_vals)
print("GPT-5 Default BERT Similarities:", GPT5_default_bert_vals)
print("GPT-4 Default BERT Similarities:", GPT4_default_bert_vals)

GPT5CffExplanations = [
    """This script implements a full big-integer arithmetic engine and then builds an RSA cryptosystem on top of it. It defines a BigInteger type with operations such as addition, subtraction, multiplication, division, modular exponentiation, and primality testing, and then introduces an RSAKey object that knows how to encrypt and decrypt messages using PKCS#1-style padding. The file hard-codes an RSA keypair (modulus, exponents, and CRT parameters), runs an encrypt/decrypt round-trip on a sample plaintext string, and verifies that the decrypted result matches the original message as a self-test of the implementation.""",

    """This script is a small constraint-solving system modeled on the classic DeltaBlue benchmark. It defines abstractions for strengths, variables, and different kinds of constraints, plus a planner that incrementally builds and executes plans to satisfy all constraints in a dependency graph. The planner adds and removes constraints, propagates changes through the graph in priority order, and maintains consistency among related variables, making the file effectively a performance benchmark and reference implementation for incremental constraint solvers.""",

    """This script manages a simple UI that remembers a user’s preferred background color using localStorage. It saves a hard-coded username, reads a stored color value on page load to set a Bootstrap background class on a main element, and falls back to a dark theme if nothing is stored yet. A button dynamically injects several color-choice buttons into the page, and clicking any of them updates the background class and persists the chosen color to localStorage so it is restored the next time the page is opened.""",

    """This script powers a configurable password generator embedded in a small web app. It tracks the desired password length and which character categories (symbols, numbers, uppercase letters, and lowercase letters) are enabled, builds a composite character pool accordingly, and then randomly draws characters from that pool to assemble a new password. The interface provides plus and minus buttons to adjust length, toggle buttons that visually show which character types are active, and a copy action that selects the generated password and copies it to the clipboard while briefly displaying a “copied” alert message.""",

    """This script defines a Color class in a ray-tracing namespace and provides utility operations for color arithmetic. Each Color instance holds red, green, and blue components and supports methods for adding and subtracting colors, scaling by a scalar, multiplying colors component-wise, clamping values into a valid range, measuring distances between colors, and blending or adjusting brightness. The class is designed to be used by a ray tracer or similar renderer so that all color computations and transformations are centralized in a single, reusable helper type.""",
        """This script is an implementation of the Richards benchmark, which simulates a small cooperative task scheduler to measure language and runtime performance. It defines a Scheduler that manages task control blocks, packet queues, and several task types such as idle, worker, handler, and device tasks, each with its own behavior. The benchmark constructs an initial network of tasks and packets, repeatedly runs the scheduler until there is no runnable task left, and finally checks that the number of processed packets and held tasks matches known expected values, throwing an error if the results are incorrect.""",


    """This script implements a browser-based rock–paper–scissors game with basic animations and scorekeeping. It wires up an intro screen and a match screen, fades the intro out when the user starts the game, and listens for clicks on rock, paper, or scissors buttons to record the player’s choice. For each round it randomly selects the computer’s choice, plays “shaking” animations on the hand images, updates those images to show the final selections, determines whether the player, the computer, or neither won, adjusts the corresponding score counters, and updates a text element to display the outcome of the round.""",

    """This script implements a very simple client-side todo list application. It reads text input from a form, and when the form is submitted it prevents the default page reload, checks for an empty value, and if there is text it creates a new list item containing the todo text plus a small “x” button. The new item is inserted into a list container, and clicking its “x” button calls a function that removes the corresponding list entry from the DOM, so items can be added and deleted without any server interaction.""",

    """This script builds a weather widget that uses the browser’s geolocation API together with the Dark Sky REST API (via a CORS proxy) and Skycons. When the page loads it requests the user’s current latitude and longitude, fetches the current weather for that location, and then displays the location’s timezone, a short summary, and the current temperature in the UI. It also uses Skycons to render an animated weather icon corresponding to the reported conditions and adds a click handler to the temperature display that toggles the numeric value between Fahrenheit and its Celsius conversion."""
]
GPT4CffExplanations = [
    """This JavaScript file implements a basic music player interface. It selects various DOM elements representing playback controls (like play, pause, next, previous), progress bar, and song metadata (title, artist, album art). It maintains a playlist and allows the user to play or pause audio, move to the next or previous track, and see playback progress. The script listens for user interactions, updates the current song display, and adjusts playback accordingly. Additionally, it updates the visual progress bar in real time and resets playback when a song ends. The functionality mirrors that of a simplified web-based audio player, demonstrating structured event handling, media control via the Audio API, and dynamic UI updates."""
,
"""This JavaScript file provides functionality for a Pomodoro timer — a time management tool that alternates periods of focused work with short and long breaks. The script sets up initial states for timer durations (Pomodoro, short break, long break), manages countdown logic using `setInterval`, and updates the UI accordingly. Users can start, pause, and reset the timer, and visual feedback is provided through a circular progress ring that animates as time progresses. It also includes logic to switch modes between work sessions and breaks, reflecting Pomodoro cycle behavior. Overall, the code demonstrates time-based state transitions, DOM interaction, and interval management to create a visually interactive productivity tool."""
,
"""This JavaScript file creates a dynamic interface where users can change the background color of a web page using various buttons. It initializes with a user name saved in local storage, then dynamically generates a set of color-themed buttons. When clicked, each button updates the background color of a designated `#fondo` element and stores the selected color in local storage for persistence. On reload, the previously selected color is restored. The script uses event delegation to handle button clicks efficiently and employs obfuscated variable names and string decoding to make static analysis more difficult, which suggests the code was intentionally transformed for obfuscation or compression purposes."""
,
"""
This JavaScript file implements a **customizable password generator** with a web-based user interface.
It allows users to choose which character types to include in the password (numbers, symbols, uppercase, lowercase),
specify the number of characters, and copy the generated password to the clipboard.

### Features:

1. **User Interface Setup**:
   - The script targets an HTML form (`#app`) and sets up event listeners on various buttons and input fields.
   - The character count is taken from an input field (`#numero-caracteres`), and increment/decrement buttons adjust this value.

2. **Options for Password Generation**:
   - The user can toggle four character sets via buttons:
     - Symbols (`btn-simbolos`)
     - Numbers (`btn-numeros`)
     - Uppercase letters (`btn-mayuscula`)
     - Lowercase letters (enabled by default, toggle logic omitted from minified code)
   - Toggling is visually represented by activating buttons (via class toggles).

3. **Password Generation Logic**:
   - When the 'Generate' button is clicked (`btn-generar`), the script:
     - Gathers all enabled character sets.
     - Concatenates all selected character strings into one array.
     - Randomly picks characters from this combined set based on the desired length.
     - Displays the generated password in an input field (`input-password`).

4. **Clipboard Copying**:
   - The 'Copy' button (`btn-copiar`) selects the password text field and uses `document.execCommand('copy')` to copy it to the clipboard.
   - Shows a temporary alert (`.alerta-copiado`) to indicate the password was copied.

5. **Obfuscation**:
   - The JavaScript is heavily **obfuscated**, using techniques such as:
     - Mangled variable names (e.g., `_0x2798f3`, `_0x1aae63`)
     - Indexed string decoding functions (`_0x2c1a`, `_0x3487`)
     - Random math operations and loop tricks to confuse static analysis
   - Despite this, the logic remains that of a basic password generator web tool.

### Conclusion:

The file is a **password generator with clipboard functionality**, wrapped in obfuscation likely for branding protection or mild deterrence against copying. It has no malicious behavior.
"""
,
        """This JavaScript file defines a Color class intended for use in a ray tracing engine or graphics application. It includes operations like color addition, scalar multiplication, clamping values between 0 and 1, and linear interpolation (blending). These features are essential in rendering systems that simulate lighting and color blending. The class structure supports chaining and encapsulates color math logic in a clean and extensible way, making it suitable for computer graphics or simulation codebases.""",

        """This file implements the Richards benchmark, simulating a multitasking scheduler with multiple task types (Idle, Worker, Handler, Device). Each task operates on packets and communicates via a simulated task control block system. The benchmark measures JavaScript engine performance and task scheduling capabilities. The implementation mimics operating system behavior in a controlled, testable environment and is commonly used in benchmarking suites and performance analysis tools.""",

    """This JavaScript file implements a Rock-Paper-Scissors game. It begins with a start screen and transitions into the game view upon user interaction. Players select rock, paper, or scissors, and the computer makes a random choice. Both choices are animated with a brief delay before showing the result. The winner is determined by standard game rules and the score is updated. The script uses DOM manipulation and event listeners to manage user input and interface changes. The code is obfuscated but structured around game logic, UI interactivity, and state updates.""",

    """This file is a simple to-do list application built using vanilla JavaScript. It allows users to input text and dynamically add it as a list item in the DOM. Each list item includes a delete button for removal. Input validation is performed to prevent empty submissions. The code uses basic DOM manipulation and event handling, and although obfuscated, clearly reflects typical structure found in beginner-level CRUD applications focused on form input and list rendering.""",

    """This script fetches and displays weather data using the browser’s Geolocation API to determine the user’s position. It sends a request to the Dark Sky API (via a proxy) and shows weather details like temperature, location, and summary in the UI. A click toggles the temperature unit between Fahrenheit and Celsius. It uses asynchronous fetch calls, basic temperature conversion, and DOM manipulation to provide a real-time weather dashboard. The code is obfuscated but structured around geolocation, API access, and UI updates.""",
]

GPT5_cff_cosine_vals = []
GPT4_cff_cosine_vals = []
GPT5_cff_bert_vals = []
GPT4_cff_bert_vals = []
GPT5_cff_gpt5_vals = [1, 1, 1, 1, 0, 1, 1, 1, 1]
GPT4_cff_gpt5_vals = [0, 0, 1, 1, 0, 1, 1, 1, 1]
for i in range(9):
    GPT5_cff_cosine_vals.append(CosineSimilarity(explanations[i], GPT5CffExplanations[i]))
    GPT4_cff_cosine_vals.append(CosineSimilarity(explanations[i], GPT4CffExplanations[i]))
    GPT5_cff_bert_vals.append(WebBertSim(explanations[i], GPT5CffExplanations[i]))
    GPT4_cff_bert_vals.append(WebBertSim(explanations[i], GPT4CffExplanations[i]))

print("GPT-5 CFF Cosine Similarities:", GPT5_cff_cosine_vals)
print("GPT-4 CFF Cosine Similarities:", GPT4_cff_cosine_vals)
print("GPT-5 CFF BERT Similarities:", GPT5_cff_bert_vals)
print("GPT-4 CFF BERT Similarities:", GPT4_cff_bert_vals)


GPT5DCIExplanations = [
    """This script implements a full big-integer arithmetic engine and then builds an RSA cryptosystem on top of it. It defines a BigInteger type with operations such as addition, subtraction, multiplication, division, modular exponentiation, and probabilistic primality testing, and then introduces an RSAKey object that encrypts and decrypts messages using PKCS#1-style padding. The file hard-codes an RSA keypair (modulus, exponents, and CRT parameters), runs an encrypt/decrypt round-trip on a sample plaintext string, and verifies that the decrypted result matches the original message as a self-test of the implementation.""",

    """This script is a small constraint-solving system modeled on the classic DeltaBlue benchmark. It defines abstractions for strengths, variables, and different categories of constraints along with a planner that incrementally builds and executes plans to satisfy all constraints in a dependency graph. As constraints are added and removed, the planner propagates changes through the graph in priority order, maintains consistency among related variables, and uses this process as a performance benchmark and reference implementation for incremental constraint solvers.""",

    """This script manages a simple UI that remembers a user’s preferred background color using localStorage. It stores a fixed username, reads a saved color key on page load to set a Bootstrap background class on a main element, and falls back to a dark theme if nothing has been stored yet. A control button dynamically injects several color-choice buttons into the page, and clicking any of them updates the element’s background class and persists the chosen color so the same theme is restored the next time the page is opened.""",

    """This script powers a configurable password generator embedded in a small web app. It tracks the desired password length and which character categories (symbols, numbers, uppercase letters, and lowercase letters) are enabled, constructs a combined character pool from the active categories, and then randomly draws characters from that pool to assemble a new password. The interface provides plus and minus buttons to adjust the length, toggle buttons that visually indicate which character types are active, and a copy feature that selects the generated password, copies it to the clipboard, and briefly shows a “copied” notification.""",

    """This script implements a compact JavaScript ray tracer, defining the core math and rendering engine needed to draw a simple 3D scene to a canvas. It includes vector and color classes, rays, materials (including solid and chessboard textures with gloss, reflection, and transparency), geometric shapes such as spheres and planes, light sources, a camera with position and look-at configuration, and an Engine that casts rays per pixel to compute intersections, shading, shadows, reflections, and background color. The engine iterates over pixels using the configured scene and camera, computes the final color for each ray, and paints the result onto the canvas, while also performing basic validation of the rendered scene.""",

    """This script is an implementation of the Richards benchmark, which simulates a small cooperative task scheduler to measure language and runtime performance. It defines a Scheduler that manages task control blocks, packet queues, and several task types such as idle, worker, handler, and device tasks, each with its own behavior and priority. The benchmark builds a network of tasks and packets, repeatedly runs the scheduler until there is no runnable task left, and finally checks that the number of processed packets and held tasks matches expected constants, throwing an error if the results do not match the benchmark’s reference values.""",

    """This script implements a browser-based rock–paper–scissors game with simple animations and scorekeeping. It wires up an intro screen and a match screen, fades the intro out when the user starts the game, and listens for clicks on rock, paper, or scissors buttons to record the player’s choice. For each round it randomly selects the computer’s choice, plays “shaking” animations on the hand images, updates those images to show the final selections, determines whether the player, the computer, or neither has won, adjusts the corresponding score counters, and updates a text element to display the outcome of the round.""",

    """This script implements a minimal client-side todo list application. It reads text input from a form, and when the form is submitted it prevents the default page reload, checks for an empty value, and if there is text it creates a new list item containing the todo text plus a small “x” button. The new item is inserted into a list container, and clicking its “x” button calls a function that removes the corresponding list entry from the DOM, so items can be added and deleted within the page without any backend or persistent storage.""",

    """This script builds a weather widget that uses the browser’s geolocation API together with a weather service (accessed via a CORS proxy) and Skycons for animated icons. When the page loads it requests the user’s current latitude and longitude, fetches the current weather for that location, and then displays the location’s timezone, a short summary, and the current temperature in the UI. It also uses Skycons to render an animated icon corresponding to the reported conditions and adds a click handler to the temperature area that toggles the displayed numeric value between Fahrenheit and its Celsius conversion."""
]
GPT4DCIExplanations = [
    """
This JavaScript file defines the core logic for a basic memory card matching game. The script initializes a set of card elements, shuffles them randomly using the Fisher-Yates algorithm, and manages the game's state — including card flipping, match checking, and game progression. When a user clicks on a card, it is flipped, and once two cards are flipped, the script checks if they match. If they do, they remain flipped; otherwise, they are turned back after a short delay. The game tracks matched pairs and disables input while animations are playing to prevent logic errors. Additionally, the script resets the game state on page load and dynamically updates the board. The use of modern JavaScript techniques (e.g., `querySelectorAll`, event listeners, and classList manipulation) makes this a clean, event-driven implementation of a classic browser-based matching game.
"""
,
"""
This JavaScript file defines the logic for a quiz game application that presents the user with a sequence of multiple-choice questions. Each question consists of a prompt and four answer options. When the user selects an answer, the script checks whether it is correct, updates the score accordingly, and moves to the next question. The score is tracked and displayed at the end of the quiz. The application uses arrays to store the questions and their associated options and answers. DOM manipulation is used to display the current question and handle user input. The code is straightforward and modular, with separate functions to load questions, check answers, and handle the end-of-quiz state. It is intended for use in an educational or entertainment setting to test knowledge through a simple web-based quiz interface.
"""
,
"""
This JavaScript file implements a dynamic theme selector for a web page using Bootstrap classes. When the page loads, it checks localStorage for a previously selected background color (`colorFondo`) and applies it to the main background element. It also initializes event listeners on a button that injects a set of color-themed buttons (primary, secondary, danger, success, warning) into the DOM. When any of these buttons are clicked, the background color of a designated element (`#fondo`) is updated according to the selected theme, and the new theme is stored in localStorage so it persists on page reloads. The code uses event delegation to handle button clicks and ensures that the correct Bootstrap class is applied. Overall, this script allows users to personalize the page’s look and remember their preference across sessions.
"""
,
"""
    This JavaScript file implements a password generator web application. It enables users to generate secure 
    passwords based on configurable criteria such as including symbols, numbers, uppercase letters, and lowercase letters. 
    The application provides UI controls to increase or decrease the number of characters, toggle inclusion of 
    different character types, and generate a password accordingly. The generated password is displayed in an 
    input field, and a 'copy' feature allows users to copy the password to the clipboard. The code is heavily obfuscated 
    with character arrays and minified functions, making it difficult to interpret without deobfuscation, but 
    its core logic focuses on DOM manipulation, user input handling, random character selection, and password generation.
    """,
    """
This JavaScript file implements the core functionality of a simple calculator application. It allows users to perform basic arithmetic operations such as addition, subtraction, multiplication, and division using on-screen buttons. When a button is pressed, the corresponding value or operator is appended to the display. The calculator handles clearing the screen, deleting individual characters, and computing the final result using JavaScript’s built-in `eval()` function. The script manages input validation to some extent by ensuring that multiple operators cannot be entered in succession. It uses DOM manipulation and event listeners to interact with HTML elements like buttons and display fields. The implementation is straightforward and functional, intended for embedding into a web page to provide interactive math capabilities.
"""
,
"""
This JavaScript file implements a real-time currency converter using an external API. It allows users to select a source currency, a target currency, and input an amount to convert. Upon clicking the 'convert' button, the script fetches the latest exchange rate data from the `https://cdn.jsdelivr.net/gh/fawazahmed0/currency-api@1/latest/currencies/` API. The resulting exchange rate is then used to calculate the converted amount, which is displayed on the web page. The script also updates the displayed currency codes beside the respective flag images, enhancing the UI clarity. The code uses modern web features such as `fetch`, `async/await`, dynamic DOM manipulation, and error handling to provide a responsive and user-friendly experience. This tool is useful for users needing up-to-date currency conversion without reloading the page.
"""
,
"""
    This obfuscated JavaScript file implements a simple Rock-Paper-Scissors game between a player and a computer. 
    It sets up a user interface with clickable options for the player to choose "rock", "paper", or "scissors". 
    When the player makes a selection, the computer randomly selects its own move. 
    The results are displayed using animations for each player's "hand", and a winner is declared based on the standard rules. 
    The game also updates and displays scores for both the player and the computer. 
    The obfuscation involves replacing strings with indexed array lookups and mangling variable names to make static analysis difficult.
    """,
"""
This JavaScript file implements a basic To-Do list application with obfuscated code. The application enables users to input tasks through a form field and display them as a list. When the user submits the form, the script checks if the input field is empty; if it is, a message is logged to the console indicating it's empty. Otherwise, the task is added to the list with a delete button. Clicking the delete button removes the task from the list. The script utilizes DOM manipulation methods like `insertAdjacentHTML`, `getElementById`, and `addEventListener`. Obfuscation techniques such as encoding string literals into a string array with indirect access through index mapping functions are used throughout the script to make it harder to read and analyze:contentReference[oaicite:0]{index=0}.
"""
,
"""
    This JavaScript file is a weather widget that retrieves and displays the current weather 
    information for the user's location using the Dark Sky API. It uses the browser's 
    Geolocation API to get the user's coordinates, constructs an API request to fetch weather 
    data (temperature, summary, and icon), and then displays it in the HTML page. It also 
    provides the functionality to toggle the temperature display between Fahrenheit and Celsius.
    The weather icon is dynamically updated using the Skycons library to match the current 
    conditions. Obfuscation techniques are used to obscure variable names and control flow.
    """,
]

GPT5_dci_cosine_vals = []
GPT4_dci_cosine_vals = []
GPT5_dci_bert_vals = []
GPT4_dci_bert_vals = []
GPT5_dci_gpt5_vals = [1, 1, 1, 1, 0, 1, 1, 1, 1]
GPT4_dci_gpt5_vals = [0, 0, 0, 0, 0, 1, 1, 1, 1]
for i in range(9):
    GPT5_dci_cosine_vals.append(CosineSimilarity(explanations[i], GPT5DCIExplanations[i]))
    GPT4_dci_cosine_vals.append(CosineSimilarity(explanations[i], GPT4DCIExplanations[i]))
    GPT5_dci_bert_vals.append(WebBertSim(explanations[i], GPT5DCIExplanations[i]))
    GPT4_dci_bert_vals.append(WebBertSim(explanations[i], GPT4DCIExplanations[i]))

GPT5SSExplanations = [
    """This script implements a full big-integer arithmetic engine and then builds an RSA cryptosystem on top of it. It defines a BigInteger type with operations such as addition, subtraction, multiplication, division, modular exponentiation, and probabilistic primality testing, and then introduces an RSAKey object that encrypts and decrypts messages using PKCS#1-style padding. The file hard-codes an RSA keypair (modulus, exponents, and CRT parameters), runs an encrypt/decrypt round-trip on a sample plaintext string, and verifies that the decrypted result matches the original message as a self-test of the implementation.""",

    """This script is a constraint-solving system modeled on the classic DeltaBlue benchmark. It defines abstractions for strengths, variables, and various kinds of constraints, together with a planner that incrementally builds and executes plans to satisfy all constraints in a dependency graph. As constraints are added and removed, the planner propagates changes through the graph in priority order and maintains consistency among related variables, making the code both a reference implementation and a performance benchmark for incremental constraint solvers.""",

    """This script manages a simple UI that remembers a user’s preferred background color using localStorage. It stores a fixed username, reads a saved color key on page load to set a Bootstrap background class on a main element, and falls back to a dark theme if nothing has been stored yet. A control button dynamically injects several color-choice buttons into the page, and clicking any of them updates the element’s background class and persists the chosen color so the same theme is restored the next time the page is opened.""",

    """This script powers a configurable password generator embedded in a small web app. It tracks the desired password length and which character categories (symbols, numbers, uppercase letters, and lowercase letters) are enabled, constructs a combined character pool from the active categories, and then randomly draws characters from that pool to assemble a new password. The interface provides plus and minus buttons to adjust the length, toggle buttons that visually indicate which character types are active, and a copy feature that selects the generated password, copies it to the clipboard, and briefly shows a “copied” notification.""",

    """This script implements a compact JavaScript ray tracer, defining the core math and rendering engine needed to draw a simple 3D scene onto a canvas. It includes vector and color classes, rays, materials (including solid and chessboard textures with gloss, reflection, and transparency), geometric shapes such as spheres and planes, light sources, a camera with position and look-at configuration, and an engine that casts rays per pixel to compute intersections, shading, shadows, reflections, and background color. The engine iterates over pixels using the configured scene and camera, computes the final color for each ray, and paints the resulting image onto the canvas, while also doing basic validation of the rendered scene.""",

    """This script is an implementation of the Richards benchmark, which simulates a small cooperative task scheduler to measure language and runtime performance. It defines a Scheduler that manages task control blocks, packet queues, and several task types such as idle, worker, handler, and device tasks, each with its own behavior and priority. The benchmark builds a network of tasks and packets, repeatedly runs the scheduler until there is no runnable task left, and finally checks that the number of processed packets and held tasks matches expected constants, throwing an error if the results do not match the benchmark’s reference values.""",

    """This script implements a browser-based rock–paper–scissors game with simple animations and scorekeeping. It wires up an intro screen and a match screen, fades the intro out when the user starts the game, and listens for clicks on rock, paper, or scissors buttons to record the player’s choice. For each round it randomly selects the computer’s choice, plays “shaking” animations on the hand images, updates those images to show the final selections, determines whether the player, the computer, or neither has won, adjusts the corresponding score counters, and updates a text element to display the outcome of the round.""",

    """This script implements a minimal client-side todo list application. It reads text input from a form, and when the form is submitted it prevents the default page reload, checks for an empty value, and if there is text it creates a new list item containing the todo text plus a small “x” button. The new item is inserted into a list container, and clicking its “x” button calls a function that removes the corresponding list entry from the DOM, so items can be added and deleted within the page without any backend or persistent storage.""",

    """This script builds a weather widget that uses the browser’s geolocation API together with a weather service (accessed via a CORS proxy) and Skycons for animated icons. When the page loads it requests the user’s current latitude and longitude, fetches the current weather for that location, and then displays the location’s timezone, a short summary, and the current temperature in the UI. It also uses Skycons to render an animated icon corresponding to the reported conditions and adds a click handler to the temperature area that toggles the displayed numeric value between Fahrenheit and its Celsius conversion."""
]
GPT4SSExplanations = [
    """
This JavaScript file defines a simple memory card game where players flip cards to match pairs. 
It begins by selecting all elements with the class "memory-card" and assigns click event listeners 
to each. When a card is clicked, it is flipped, and the game logic determines if two flipped cards match. 
If they match, the cards remain face-up; otherwise, they are flipped back over after a delay. The code ensures 
that only two cards can be flipped at a time and includes logic to prevent additional flips while a comparison 
is in progress. It also handles resetting the board state after each move and ensures the same card isn't 
matched with itself. The implementation uses simple DOM manipulation, class toggling, and internal state flags 
to manage gameplay interactions.
"""
,
"""
This JavaScript file implements a basic image carousel or slider functionality for a web page. 
It allows users to navigate between different slides (images or content sections) by clicking 
"next" or "previous" buttons. The script keeps track of the current slide index and updates the 
active slide accordingly by toggling the "active" class on the appropriate HTML elements. 
The design ensures that navigation wraps around — if the user is on the last slide and clicks 
"next", they are taken to the first slide, and vice versa for the "previous" button. 
The code uses standard DOM methods like `querySelector`, `querySelectorAll`, and 
`addEventListener` for interaction handling, and it updates slide visibility through class manipulation.
"""
,
 """
    This JavaScript file implements a dynamic button-based background color changer for a web page.
    It uses obfuscated code and string arrays to make the code harder to analyze directly. Upon loading,
    it retrieves a user's name and preferred background color from `localStorage`, then sets the page's
    background accordingly. It also defines a UI with multiple styled buttons ("Primary", "Secondary",
    "Success", "Danger", "Warning") which, when clicked, change the background color of an element with
    ID `fondo` and save the selected color to `localStorage`. The buttons are dynamically injected into 
    the DOM by clicking another button with ID `btnBotones`. Event delegation is used to handle color 
    change logic based on the button clicked. The design demonstrates dynamic DOM manipulation, localStorage 
    persistence, and background color switching using CSS classes.
    """,
"""
This JavaScript file implements a **secure random password generator** interface using DOM manipulation
and obfuscated variable names.

Key Features:
-------------
1. **Password Generation UI**:
   - The script initializes a web form that lets the user configure password generation settings
     such as:
     - Length of the password
     - Inclusion of symbols, numbers, uppercase and lowercase letters

2. **User Interaction**:
   - Button handlers allow users to:
     - Increase or decrease the password length
     - Toggle inclusion of symbols, numbers, uppercase letters
     - Generate a new password based on current settings
     - Copy the generated password to clipboard

3. **Password Construction Logic**:
   - Character pools are defined as space-separated strings for different categories.
   - Based on toggled options, these pools are concatenated and split into an array.
   - The password is generated by randomly selecting characters from the combined pool.

4. **Clipboard Interaction**:
   - Uses `execCommand('copy')` to copy the generated password to clipboard.
   - Shows a temporary visual confirmation message when the copy occurs.

5. **Obfuscation**:
   - The entire file is heavily obfuscated using array indirection and meaningless variable names
     for string literals, making it harder to read manually.
   - This includes dynamic function names, array lookups, and Unicode character tricks.

Purpose:
--------
To provide a compact and interactive password generation tool for users via a web UI, while obscuring
the internal logic to deter reverse engineering.
"""
,
"""
This JavaScript file contains an obfuscated implementation of a ray tracer engine, which is a rendering algorithm used to simulate the behavior of light in a 3D scene. The code is heavily obfuscated, making it difficult to read, but its functionality can be inferred as follows:

1. **Ray Tracer Engine**:
   - The file defines a `Flog.RayTracer` namespace that contains various classes and methods for ray tracing.
   - It includes classes for `Color`, `Vector`, `Light`, `Material`, `Shape`, `Sphere`, `Plane`, and `Scene`.
   - These classes work together to simulate the interaction of light with objects in a 3D environment.

2. **Rendering Process**:
   - The `Engine` class is responsible for rendering the scene. It takes a `Scene` object as input and iterates over each pixel of the canvas to compute the color based on ray-object intersections.
   - The engine supports features like diffuse rendering, shadows, highlights, and reflections, which can be toggled via configuration options.

3. **Scene Setup**:
   - The `renderScene` function sets up a sample 3D scene with:
     - A camera positioned in the scene.
     - A background with ambient lighting.
     - Multiple shapes, including spheres and planes, with different materials (e.g., solid colors, chessboard patterns).
     - Lights to illuminate the scene.
   - The function then uses the `Engine` class to render the scene.

4. **Benchmarking**:
   - A `Benchmark` class is defined to repeatedly render the scene multiple times (5 iterations) to measure the performance of the ray tracer.

5. **Obfuscation**:
   - The code is obfuscated using techniques like variable renaming, string splitting, and function aliasing, making it challenging to read and understand.
   - For example, functions and variables are named with cryptic identifiers, and string values are dynamically reconstructed at runtime.

Overall, this file implements a basic ray tracing engine for rendering 3D scenes, with support for lighting, materials, and reflections. The obfuscation is likely intended to make the code harder to reverse engineer or analyze.
""",
"""
This JavaScript file implements the Richards benchmark, which simulates an operating-system-like
task dispatcher. The benchmark is designed to measure the performance of task scheduling and
message passing between tasks. It includes the following components:

1. **Scheduler**:
   - Manages a fixed set of task control blocks and packet queues.
   - Schedules tasks based on their priorities and states (e.g., running, suspended, held).
   - Provides methods to add different types of tasks (idle, worker, handler, device) and to
     manage the execution of tasks.

2. **Task Types**:
   - **IdleTask**: A task that performs no meaningful work but alternates between releasing
     packets to other tasks.
   - **WorkerTask**: A task that processes packets and alternates between two handlers.
   - **HandlerTask**: A task that processes incoming packets and performs operations based on
     their type (work or device).
   - **DeviceTask**: A task that simulates a device by processing packets or waiting for new ones.

3. **Packet**:
   - Represents a unit of communication between tasks.
   - Contains fields for the packet's ID, type, and data.

4. **TaskControlBlock**:
   - Represents a task in the scheduler.
   - Tracks the task's state, priority, and associated packet queue.

5. **Benchmark Execution**:
   - The `runRichards` function sets up the scheduler with a predefined network of tasks and
     packets.
   - Tasks are added to the scheduler, and the simulation runs until all tasks are completed.
   - The benchmark verifies the correctness of the execution by comparing the queue and hold
     counts against expected values.

6. **Obfuscation**:
   - The code is heavily obfuscated, with variable names and function names replaced by
     meaningless identifiers. This makes the code harder to read and understand.

Overall, this file is a performance benchmark that stresses task scheduling, priority-based
execution, and message passing between tasks. It is not intended for practical use but rather
to measure computational efficiency in a controlled environment.
""",
"""
This JavaScript file implements a browser-based rock-paper-scissors game with animations and score tracking. 
The game logic and UI interactions are obfuscated, but the following functionality can be inferred:

1. **Game Initialization**:
   - The `game` function initializes the game state, including player and computer scores.
   - It sets up event listeners for UI elements and handles animations.

2. **UI Elements**:
   - The game uses DOM elements for the intro screen, match screen, player/computer hands, and score display.
   - Buttons allow the player to choose between "rock," "paper," and "scissors."

3. **Game Logic**:
   - The computer's choice is randomly selected from "rock," "paper," or "scissors."
   - The player's choice is determined by the button they click.
   - The game compares the player's choice with the computer's choice to determine the winner:
     - If the choices are the same, it's a tie.
     - Otherwise, the winner is determined based on the rules of rock-paper-scissors.

4. **Animations**:
   - Hand images for both the player and computer are animated with a "shake" effect before revealing their choices.
   - The animations are reset after each round.

5. **Score Tracking**:
   - The player's and computer's scores are updated and displayed after each round.
   - The scores are stored in variables and updated in the DOM.

6. **Obfuscation**:
   - The code is heavily obfuscated, with variable names and function names replaced by unreadable identifiers.
   - A decoding function (`_0x3fd1`) maps obfuscated strings to their actual values.

7. **Execution**:
   - The `game` function is called at the end of the file to start the game.

Overall, this file provides a simple interactive rock-paper-scissors game with animations and score tracking, designed to run in a web browser.
""",
"""
This JavaScript file implements a simple to-do list application with basic add and remove functionality. 
The code is obfuscated, but its functionality can be inferred as follows:

1. **Form Submission Handling**:
   - The `todoForm` element listens for the "submit" event.
   - When the form is submitted, the default behavior (e.g., page reload) is prevented.
   - If the input field (`todoInput`) is empty, a helper function `inputoEmpty()` logs an "empty" message to the console.
   - If the input is not empty, the `addItem()` function is called with the input value.

2. **Adding Items**:
   - The `addItem()` function creates a new list item (`<li>`) containing the entered text and an inline delete button.
   - The delete button is configured to call the `removeItem()` function when clicked.
   - The new list item is inserted at the top of the `itemList` container.
   - After adding the item, the input field is cleared and refocused.

3. **Removing Items**:
   - The `removeItem()` function removes the corresponding list item from the DOM when the delete button is clicked.

4. **Obfuscation**:
   - The code is heavily obfuscated, with variable names and function names replaced by dynamically resolved strings.
   - A mapping function (`_0x4c03`) and an array of encoded strings (`_0x2fd9`) are used to decode the obfuscated names at runtime.

Overall, this script provides a minimal to-do list application with the ability to add and remove items dynamically.
""",
"""
This JavaScript file implements a geolocation-based weather widget. The code is obfuscated, 
but its functionality can be inferred as follows:

1. **Geolocation Retrieval**:
   - The script uses the browser's `navigator.geolocation` API to retrieve the user's current latitude and longitude.

2. **Weather Data Fetching**:
   - The latitude and longitude are used to construct a URL for a weather API (likely Dark Sky or a similar service).
   - The script fetches weather data in JSON format, which includes details such as temperature, weather summary, and an icon representing the current weather.

3. **DOM Manipulation**:
   - The script updates various DOM elements to display the retrieved weather information:
     - The timezone is displayed in a specific element.
     - The temperature is shown in another element.
     - A weather summary (e.g., "Sunny", "Cloudy") is displayed in a description element.
     - A weather icon is rendered using the Skycons library.

4. **Temperature Unit Toggle**:
   - The script allows toggling between Fahrenheit and Celsius for the temperature display.
   - A click event listener is added to a button or element, which switches the temperature unit and updates the displayed value accordingly.

5. **Skycons Integration**:
   - The Skycons library is used to display animated weather icons. The icon type is determined from the weather data and mapped to the appropriate Skycons icon.

6. **Obfuscation**:
   - The code is heavily obfuscated, with variable names and function names replaced by nonsensical strings.
   - A decoding function (`_0xb7bd`) is used to map obfuscated strings to their actual values at runtime.

Overall, this file provides a weather widget that dynamically fetches and displays weather information based on the user's location, with support for animated icons and temperature unit toggling.
"""
]
GPT5_ss_cosine_vals = []
GPT4_ss_cosine_vals = []
GPT5_ss_bert_vals = []
GPT4_ss_bert_vals = []
GPT5_ss_gpt5_vals = [1, 1, 1, 1, 0, 1, 1, 1, 1]
GPT4_ss_gpt5_vals = [0, 0, 0, 0, 0, 1, 1, 1, 1]
for i in range(9):
    GPT5_ss_cosine_vals.append(CosineSimilarity(explanations[i], GPT5SSExplanations[i]))
    GPT4_ss_cosine_vals.append(CosineSimilarity(explanations[i], GPT4SSExplanations[i]))
    GPT5_ss_bert_vals.append(WebBertSim(explanations[i], GPT5SSExplanations[i]))
    GPT4_ss_bert_vals.append(WebBertSim(explanations[i], GPT4SSExplanations[i]))

GPT5WOExplanations = [
    """This script implements a full big-integer arithmetic engine and then builds an RSA cryptosystem on top of it. It defines a BigInteger type with operations such as addition, subtraction, multiplication, division, modular exponentiation, and probabilistic primality testing, and then introduces an RSAKey object that encrypts and decrypts messages using PKCS#1-style padding. The file hard-codes an RSA keypair (modulus, exponents, and CRT parameters), runs an encrypt/decrypt round-trip on a sample plaintext string, and verifies that the decrypted result matches the original message as a self-test of the implementation.""",

    """This script is a constraint-solving system modeled on the classic DeltaBlue benchmark. It defines abstractions for strengths, variables, and various kinds of constraints, together with a planner that incrementally builds and executes plans to satisfy all constraints in a dependency graph. As constraints are added and removed, the planner propagates changes through the graph in priority order, maintains consistency among related variables, and uses this process as both a reference implementation and a performance benchmark for incremental constraint solvers.""",

    """This script manages a simple UI that remembers a user’s preferred background color using localStorage. It stores a fixed username, reads a saved color key on page load to set a Bootstrap background class on a main element (falling back to a dark theme if nothing is stored), and wires a button that dynamically injects several color-choice buttons into the page. Clicking any of these buttons updates the element’s background class to the corresponding Bootstrap color utility and persists the chosen color so the same theme is restored the next time the page is opened.""",

    """This script powers a configurable password generator embedded in a small web app. It tracks the desired password length and which character categories (symbols, numbers, uppercase letters, and lowercase letters) are enabled, constructs a combined character pool from the active categories, and then randomly draws characters from that pool to assemble a new password string. The interface provides plus and minus buttons to adjust the length, toggle buttons that visually indicate which character types are active, and a copy feature that selects the generated password, copies it to the clipboard, and briefly shows a “copied” notification banner to the user.""",

    """This script defines a Color class within a ray-tracing namespace and provides utility operations for color arithmetic. Each Color instance holds red, green, and blue components and supports methods for adding and subtracting colors, blending between two colors, multiplying colors component-wise or by a scalar, dividing by a scalar, clamping component values into the valid range, and computing a simple distance or brightness measure. The class is intended as a reusable helper for a ray tracer or other rendering code so that all color computations and transformations are encapsulated in a single, consistent abstraction.""",

    """This script is an implementation of the Richards benchmark, which simulates a small cooperative task scheduler to measure language and runtime performance. It defines a Scheduler that manages task control blocks, packet queues, and several task types such as idle, worker, handler, and device tasks, each with its own behavior and priority. The benchmark constructs an initial network of tasks and packets, repeatedly runs the scheduler until there is no runnable task left, and then checks that the number of processed packets and held tasks matches expected constants, throwing an error if the results deviate from the benchmark’s reference values.""",

    """This script implements a browser-based rock–paper–scissors game with simple animations and scorekeeping. It wires up an intro screen and a match screen, fades the intro out when the user clicks the play button, and listens for clicks on rock, paper, or scissors buttons to record the player’s choice. For each round it randomly selects the computer’s choice, applies “shake” animations to the hand images, updates those images to show the chosen gestures, determines whether the player, the computer, or neither has won, increments the corresponding score counters, and updates a text element to display the outcome of the round.""",

    """This script implements a minimal client-side todo list application. It reads text input from a form and, on submission, prevents the default page reload, checks whether the input is empty, and if not creates a new list item containing the todo text plus a small “x” button. The new item is inserted into a list container, and clicking its “x” button removes the corresponding list entry from the DOM, allowing items to be added and deleted dynamically in the page without any backend or persistent storage logic.""",

    """This script builds a weather widget that uses the browser’s geolocation API together with the Dark Sky weather service (accessed via a CORS proxy) and Skycons for animated icons. When the page loads it requests the user’s current latitude and longitude, fetches the current weather for that location, and then displays the location’s timezone, a short textual summary, and the current temperature in the UI. It also draws an appropriate animated icon for the reported conditions and attaches a click handler to the temperature section that toggles the displayed units between Fahrenheit and Celsius by converting the value on the fly."""
]
GPT4WOExplanations = [
    """
This JavaScript file is an obfuscated script that implements functionality for injecting
external JavaScript code into a webpage from a hardcoded remote URL. It uses several 
techniques to hide its intent, such as wrapping operations in anonymous functions, 
using encoded strings, and dynamically constructing DOM elements and method calls. 
The core behavior revolves around creating a new `<script>` element with its `src` 
attribute set to a specific external link (in this case, a Discord CDN URL hosting a JS 
payload), and appending it to the current document's `<head>`. This effectively causes the 
browser to fetch and execute additional code hosted remotely. This behavior is typical of 
JavaScript-based loaders or droppers used in malicious contexts to offload the main payload 
and avoid detection in static analysis.
"""
,
 """
    This JavaScript file implements a version of the DeltaBlue constraint-solving algorithm, which is used for maintaining relationships between variables in a system of constraints. 
    The code defines several key components:
    
    - A `Strength` hierarchy for managing constraint priority.
    - `Variable` objects that represent values subject to constraints.
    - `Constraint` classes, including `UnaryConstraint`, `StayConstraint`, `EditConstraint`, `BinaryConstraint`, `ScaleConstraint`, and `EqualityConstraint`, each modeling different types of relationships.
    - An `OrderedCollection` class for managing lists of constraints or variables.
    - A `Planner` class that incrementally adds and removes constraints, maintaining a plan of how constraints should be executed in dependency order.
    - A `Plan` class that stores and executes ordered constraints.
    
    Two test functions—`chainTest(n)` and `projectionTest(n)`—build test scenarios involving chains of equality constraints and projection transformations respectively. 
    These verify the solver's ability to propagate changes across variable networks.
    
    The file also includes embedded WebAssembly code to obfuscate certain string values, and a `Benchmark` class that runs `deltaBlue()` (which performs the two tests) 20 times in a loop. 
    
    Overall, the file serves both as an implementation of the DeltaBlue algorithm and a performance benchmark suite for constraint-solving in JavaScript.
    """,
     """
    This JavaScript file implements a simple dynamic theming system using WebAssembly (via a minimal runtime called 
    "Wobfuscator-lite") to obfuscate and decode string constants. At its core, it defines a WebAssembly module 
    that exposes two functions: `memory` and `getOffset`. These are used to locate strings in memory dynamically 
    through the `__wstr__` function. The script retrieves and stores a user-defined name in `localStorage`, applies 
    a background color based on a stored preference, and allows the user to dynamically generate a set of color-coded 
    buttons. Clicking one of these buttons changes the background color of the page and stores this preference. The 
    event delegation mechanism ensures the color change logic is centralized and responsive. The use of WebAssembly 
    for string obfuscation indicates a basic attempt to hinder static analysis or tampering of the JavaScript code.
    """,
     """
    This JavaScript file implements a password generator web application, using a combination of WebAssembly (Wasm) and DOM manipulation. The file begins by decoding a base64-encoded WebAssembly binary that provides string obfuscation via index-based lookup (`__wstr__`). It then initializes a configuration object that controls the character types to include in generated passwords—such as symbols, numbers, uppercase, and lowercase letters—and sets the desired password length from user input. The script attaches event listeners to buttons for increasing/decreasing password length and toggling inclusion of each character type. When the "generate" button is clicked, a password is created from the selected character pools and displayed in an output field. Another button allows users to copy the generated password to the clipboard, with visual feedback indicating success. Overall, the code obfuscates string literals via a Wasm runtime and provides a customizable, interactive password generation UI in the browser.
    """,
     """
    This JavaScript file is a basic stopwatch/timer application that tracks time in seconds, minutes, and hours.
    It uses WebAssembly (WASM) for obfuscating string literals via a `__wstr__` function which retrieves strings
    from a linear memory buffer. The core functionality initializes and manipulates a display element that shows 
    elapsed time in the format hh:mm:ss. The code sets up a `setInterval` timer that increments the seconds, 
    rolling over to minutes and hours as needed, and updates the display every second. Users can start, stop, 
    and reset the timer using buttons that are wired to event listeners. The use of WASM for simple string 
    obfuscation provides a lightweight anti-reverse-engineering mechanism, but the overall functionality is 
    straightforward and geared toward tracking elapsed time within a web interface.
    """,
     """
    This JavaScript file implements the Richards benchmark, which simulates an operating system-like task scheduler 
    with multiple tasks interacting via message packets. It includes the Wobfuscator-lite runtime, a WebAssembly-based 
    string obfuscation mechanism used to hide certain string constants in the code. The runtime decodes obfuscated strings 
    at runtime using WebAssembly.

    The core logic sets up a scheduler with several tasks: IdleTask, WorkerTask, HandlerTask, and DeviceTask. These tasks 
    are added to the scheduler with different priorities and packet queues. The scheduler simulates task execution using 
    cooperative multitasking and message passing. Tasks manipulate packets, change states, and simulate work, such as processing 
    data or interacting with devices. The benchmark ensures the simulated system reaches expected task queue and hold counts, 
    throwing an error if these are not achieved.

    The code is structured in a modular object-oriented style, using prototypes for task behaviors and a control block structure 
    for managing execution. The obfuscation mechanism further complicates direct string analysis, making it harder to reverse-engineer 
    the runtime messages.
    """,
    """
    This JavaScript file implements a simple Rock-Paper-Scissors game with a graphical interface using DOM manipulation, event listeners, and animations. 
    It uses a small embedded WebAssembly (WASM) runtime (from a tool called "Wobfuscator-lite") to obfuscate string literals like element selectors, event types, and game messages. 
    The WASM module exposes a function to retrieve string offsets from memory, making it harder to directly read game strings in the source code.

    The game logic is divided into several parts:
    - `startGame` sets up the initial screen and transitions to the match screen when the play button is clicked.
    - `playMatch` sets up the user interaction, retrieves player choices, randomly selects a computer move, and updates the visuals with animations.
    - `compareHands` contains the logic for determining the winner based on standard Rock-Paper-Scissors rules and updates the scores accordingly.
    - `updateScore` updates the displayed score for both the player and the computer.

    The obfuscation mechanism helps obscure static strings used in selectors and gameplay messages, making reverse engineering or inspection slightly more difficult.
    """,
    """
    This JavaScript file implements a lightweight to-do list application using a WebAssembly-based string obfuscation runtime called "Wobfuscator-lite". The WebAssembly module stores obfuscated strings, and functions like `__wstr__` are used to decode and retrieve these strings at runtime, thereby making the code more resistant to static analysis or reverse engineering. The main application logic creates a simple UI interaction: when a form is submitted, it adds an item to a list unless the input is empty, in which case it logs an "empty input" warning. Items can also be removed via a dynamically attached button. All DOM element IDs and string literals are obfuscated through the WebAssembly decoding mechanism.
    """,
    """
    This JavaScript file implements a weather widget that displays real-time weather data using the Dark Sky API.
    It uses WebAssembly (via a 'Wobfuscator-lite' runtime) to obfuscate and decode string constants for security or anti-reverse-engineering purposes.
    The script first checks if geolocation is available in the user's browser. If it is, it fetches the user's latitude and longitude,
    constructs a request to the Dark Sky API through a proxy, and retrieves current weather information such as temperature, summary,
    and icon representing the weather condition. It then updates various DOM elements to display the location's timezone, current temperature,
    and weather description. The widget allows toggling between Fahrenheit and Celsius by clicking on the temperature section.
    Skycons is used to render animated weather icons corresponding to the current weather.
    """
]

GPT5_wo_cosine_vals = []
GPT4_wo_cosine_vals = []
GPT5_wo_bert_vals = []
GPT4_wo_bert_vals = []
GPT5_wo_gpt5_vals = [1, 1, 1, 1, 0, 1, 1, 1, 1]
GPT4_wo_gpt5_vals = [0, 1, 1, 1, 0, 1, 1, 1, 1]

for i in range(9):
    GPT5_wo_cosine_vals.append(CosineSimilarity(explanations[i], GPT5WOExplanations[i]))
    GPT4_wo_cosine_vals.append(CosineSimilarity(explanations[i], GPT4WOExplanations[i]))
    GPT5_wo_bert_vals.append(WebBertSim(explanations[i], GPT5WOExplanations[i]))
    GPT4_wo_bert_vals.append(WebBertSim(explanations[i], GPT4WOExplanations[i]))

GPT5generateMetrics = [1, 1, 1, 1, 1]
GPT4generationMetrics = [1, 1, 1, 1, 0]
GPT5compileMetrics = [0, 1, 0, 1, 1]
GPT4compileMetrics = [1, 1, 1, 0, 0]
GPT5correctoutputMetrics = [0, 0, 0, 1, 1]
GPT4correctoutputMetrics = [0, 0, 1, 0, 0]
# -------------------------------
# 1. Helper to compute averages
# -------------------------------
def avg(values):
    return sum(values) / len(values) if values else 0.0

# WO currently uses default explanations as a proxy

# -------------------------------
# 2. Build method-wise averages
# -------------------------------
methods = ["DE", "DE+DI", "DE+CFF", "DE+SS", "WO"]

# ----- Cosine -----
cosine_gpt5 = [
    avg(GPT5_default_cosine_vals),   # DE
    avg(GPT5_dci_cosine_vals),       # DE+DI
    avg(GPT5_cff_cosine_vals),       # DE+CFF
    avg(GPT5_ss_cosine_vals),        # DE+SS
    avg(GPT5_wo_cosine_vals),        # WO
]

cosine_gpt4 = [
    avg(GPT4_default_cosine_vals),   # DE
    avg(GPT4_dci_cosine_vals),       # DE+DI
    avg(GPT4_cff_cosine_vals),       # DE+CFF
    avg(GPT4_ss_cosine_vals),        # DE+SS
    avg(GPT4_wo_cosine_vals),        # WO
]

# ----- BERT -----
bert_gpt5 = [
    avg(GPT5_default_bert_vals),     # DE
    avg(GPT5_dci_bert_vals),         # DE+DI
    avg(GPT5_cff_bert_vals),         # DE+CFF
    avg(GPT5_ss_bert_vals),          # DE+SS
    avg(GPT5_wo_bert_vals),          # WO
]

bert_gpt4 = [
    avg(GPT4_default_bert_vals),     # DE
    avg(GPT4_dci_bert_vals),         # DE+DI
    avg(GPT4_cff_bert_vals),         # DE+CFF
    avg(GPT4_ss_bert_vals),          # DE+SS
    avg(GPT4_wo_bert_vals),          # WO
]

gpt5_gpt5 = [
    avg(GPT5_default_gpt5_vals),     # DE
    avg(GPT5_dci_gpt5_vals),         # DE+DI
    avg(GPT5_cff_gpt5_vals),         # DE+CFF
    avg(GPT5_ss_gpt5_vals),          # DE+SS
    avg(GPT5_wo_gpt5_vals),          # WO
]
gpt5_gpt4 = [
    avg(GPT4_default_gpt5_vals),     # DE
    avg(GPT4_dci_gpt5_vals),         # DE+DI
    avg(GPT4_cff_gpt5_vals),         # DE+CFF
    avg(GPT4_ss_gpt5_vals),          # DE+SS
    avg(GPT4_wo_gpt5_vals),          # WO
]

gpt5_compile_run_correct = [
    avg(GPT5generateMetrics),
    avg(GPT5compileMetrics),
    avg(GPT5correctoutputMetrics),
]
gpt4_compile_run_correct = [
    avg(GPT4generationMetrics),
    avg(GPT4compileMetrics),
    avg(GPT4correctoutputMetrics),
]

# -------------------------------
# 3. Plot cosine similarity chart
# -------------------------------
x = np.arange(len(methods))  # positions for each obfuscation method
width = 0.35                 # bar width

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x - width/2, cosine_gpt5, width, label="GPT-5")
ax.bar(x + width/2, cosine_gpt4, width, label="GPT-4o")

ax.set_xlabel("Obfuscation Method")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Cosine Similarity by Obfuscation Method and Model")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)

fig.tight_layout()
plt.savefig("cosine_similarity_by_method.png", dpi=300)
plt.show()

# -------------------------------
# 4. Plot BERT similarity chart
# -------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.bar(x - width/2, bert_gpt5, width, label="GPT-5")
ax2.bar(x + width/2, bert_gpt4, width, label="GPT-4o")

ax2.set_xlabel("Obfuscation Method")
ax2.set_ylabel("BERT Similarity")
ax2.set_title("BERT Similarity by Obfuscation Method and Model")
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.legend()
ax2.grid(axis="y", linestyle="--", alpha=0.3)

fig2.tight_layout()
plt.savefig("bert_similarity_by_method.png", dpi=300)
plt.show()

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.bar(x - width/2, gpt5_gpt5, width, label="GPT-5")
ax3.bar(x + width/2, gpt5_gpt4, width, label="GPT-4o")
ax3.set_xlabel("Obfuscation Method")
ax3.set_ylabel("GPT-5 based Similarity")
ax3.set_title("GPT-5 based Similarity by Obfuscation Method and Model")
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.legend()
ax3.grid(axis="y", linestyle="--", alpha=0.3)
fig3.tight_layout()
plt.savefig("gpt5_vs_gpt4o_agreement_by_method.png", dpi=
300)
plt.show()

fig4, ax4 = plt.subplots(figsize=(8, 5))
labels = ["Generation", "Compilation", "Correct Output"]
x2 = np.arange(len(labels))
ax4.bar(x2 - width/2, gpt5_compile_run_correct, width, label="GPT-5")
ax4.bar(x2 + width/2, gpt4_compile_run_correct, width, label="GPT-4o")
ax4.set_xlabel("Metrics")
ax4.set_ylabel("Proportion")
ax4.set_title("Code Generation, Compilation, and Correct Output Rates")
ax4.set_xticks(x2)
ax4.set_xticklabels(labels)
ax4.legend()
ax4.grid(axis="y", linestyle="--", alpha=0.3)
fig4.tight_layout()
plt.savefig("generation_compilation_correct_output_rates.png", dpi=300)
plt.show()

# for filename in expected_order:
#     file_path = os.path.join(DE_FOLDER_PATH, filename)
#     if os.path.isdir(file_path):
#         continue
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             content = f.read()
#     except Exception as e:
#         print(f"Skipping {filename} (error reading file: {e})")
#         continue

#     prompt = f"Analyze and tell me what this file does. Code: {content}"
#     prompts.append(prompt)

# GPT35TurboMetrics = GPT35TurboAnalysis(prompts, explanations)
# print(GPT35TurboMetrics)
# GPT4Metrics = GPT4Analysis(prompts, explanations)

# def GPT35TurboDeobfuscation(filepath):
#     response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "user",
#             "content": f"““You are an expert in code analysis. De-obfuscate the code and generate a readable new version. Code: {filepath}"
#         }
#     ]
#     )
#     return response.choices[0].message.content

# def GPT41Deobfuscation(filepath):
#     response = client.chat.completions.create(
#     model="gpt-4.1",
#     messages=[
#         {
#             "role": "user",
#             "content": f"““You are an expert in code analysis. De-obfuscate the code and generate a readable new version. Code: {filepath}"
#         }
#     ]
#     )
#     return response.choices[0].message.content

# # The parameter filepaths should contain all the IOCCC files

# def deobfuscationMetrics(filepaths):
#     gpt4generated = 0
#     gpt35generated = 0
#     gpt4Compile = 0
#     gpt35Compile = 0
#     for currentFile in filepaths:
#         gpt35code = GPT35TurboDeobfuscation(currentFile)
#         gpt4code = GPT41Deobfuscation(currentFile)
#         if ("```" in gpt35code):
#             gpt35generated += 1
#             match = re.search(r'```(.*?)```', gpt35code, re.DOTALL)
#             if match:
#                 code = match.group(1).strip()
#                 if (checkCompile(code)):
#                     gpt35Compile += 1
#         if ("```" in gpt4code):
#             gpt4generated += 1
#             match = re.search(r'```(.*?)```', gpt4code, re.DOTALL)
#             if match:
#                 code = match.group(1).strip()
#                 if (checkCompile(code)):
#                     gpt4Compile += 1
#     gpt35generation = gpt35generated / len(filepaths)
#     gpt35Compilation = gpt35Compile / len(filepaths)
#     gpt4generation = gpt4generated / len(filepaths)
#     gpt4Compilation = gpt4Compile / len(filepaths)

#     return (gpt35generation, gpt4generation, gpt35Compilation, gpt4Compilation)

# # This checkCompile function only checks if c code is compilable. 
# # For other languages, we can easily extend this functionality by first check if the string Python/C/cpp exists in the API response and feeding the code to the respective compiler function. 
# def checkCompile(code):
#     with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
#         f.write(code)
#         c_file_path = f.name
    
#     try:
#         result = subprocess.run(
#             ['gcc', c_file_path, '-o', "program"],
#             capture_output=True,
#             text=True,
#             timeout=10
#         )
        
#         # Check if compilation was successful
#         return (result.returncode == 0)
#     finally:
#         # Clean up the temporary C file
#         if os.path.exists(c_file_path):
#             os.remove(c_file_path)
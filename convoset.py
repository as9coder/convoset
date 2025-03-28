import tkinter as tk
from tkinter import scrolledtext, messagebox
import json
import requests
import threading
import time
import random
import re # For word counting

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
MODEL_NAME = "mistral"  # Or whichever model you have pulled in Ollama
FILE_NAME = "daily_conversations_data.jsonl" # << UPDATED Filename
END_OF_TEXT_TOKEN = "<|endoftext|>"

# --- Generation Constraints (Define BEFORE use in META_PROMPT) ---
MAX_PROMPT_WORDS = 100
MAX_RESPONSE_WORDS = 200
# Estimate tokens needed for max words (adjust if needed, ~1.3-1.5 tokens/word)
RESPONSE_NUM_PREDICT = int(MAX_RESPONSE_WORDS * 1.6) # Give buffer
USER_QUERY_NUM_PREDICT = int(MAX_PROMPT_WORDS * 1.6) # Buffer for user query too

# --- Content Generation Agenda ---
# SOLE FOCUS: Meaningful, everyday human-like conversations (generic, no names)
# Instruction: Simple, relatable daily topics.

# --- META_PROMPT - *** REWRITTEN FOR CONVERSATION FOCUS *** ---
META_PROMPT = f"""Generate a single, unique conversational opening, question, or statement suitable for a large language model, simulating one side of a typical, everyday human conversation.

**Focus EXCLUSIVELY on:**
*   **Everyday Topics:** Weather, simple daily plans or activities (like cooking, commuting, relaxing), general well-being, opinions on common/current (but not obscure) events, simple observations about surroundings, discussing general hobbies or interests (like reading, watching movies, walking - keep it generic).
*   **Conversational Tone:** Phrasing should sound natural, like someone casually talking to another person. It can be a question, a statement inviting a response, or sharing a simple thought/feeling.

**Examples:**
*   "The weather forecast looks pretty good for the weekend, doesn't it?"
*   "Just got back from a walk, it felt nice to get some fresh air."
*   "Thinking about what to make for dinner tonight, any simple ideas?"
*   "Did you see that interesting documentary about nature last night?" (Avoid *specific* obscure titles)
*   "I've been trying to read more lately, it's quite relaxing."
*   "How was your commute today?"
*   "Feeling a bit tired today, looking forward to relaxing later."
*   "What's a small thing that made you smile recently?"

**IMPORTANT RULES:**
*   **Strictly Conversational:** Do NOT generate requests for definitions, explanations, summaries, creative writing prompts, code, or anything that isn't part of a natural, everyday chat.
*   **Simplicity:** Keep the language and concepts simple and relatable. AVOID technical jargon, deep philosophical debates, complex problem-solving, or highly specialized topics.
*   **No Specific Names:** Absolutely NO personal names (like John, Maria, etc.). Use generic terms like 'someone', 'a friend', 'people' ONLY if essential, but prefer phrasing that doesn't require them.
*   **Meaningful & Realistic:** The conversation starter should make sense in a daily context and invite a sensible, conversational reply. Avoid nonsensical or overly strange statements.
*   **Length:** The generated text MUST NOT exceed {MAX_PROMPT_WORDS} words.
*   **Variety:** Do not repeat text you have generated previously in this session.
*   **Output:** Output ONLY the conversational text itself, with no preamble, labels, or explanation.
"""
# --- End META_PROMPT ---


MIN_DELAY_SECONDS = 2 # Minimum delay between generation cycles
MAX_DELAY_SECONDS = 5 # Maximum delay between generation cycles
OLLAMA_TIMEOUT = 180 # Timeout for Ollama calls
# --- End Configuration ---

# Set to store existing prompt-response pairs {(full_prompt_string, response_with_token)} for duplicate checking
existing_pairs = set()
# Set to store user queries generated *in this session* to encourage variety from the meta-prompt
session_user_queries = set()

# Flag to control the automation loop
automation_running = False
saved_count = 0

# --- Functions (Keep existing: count_words, truncate_to_words, contains_specific_name, COMMON_NAMES, load_existing_data, update_status, update_saved_count_label, call_ollama, save_pair_to_jsonl) ---

def count_words(text):
    """Counts words in a string, handling basic punctuation."""
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def truncate_to_words(text, max_words):
    """Truncates text to a maximum number of words."""
    words = text.split()
    if len(words) > max_words:
        # Try to truncate at a sentence boundary if possible near the limit
        truncated_text = ' '.join(words[:max_words])
        last_punc = max(truncated_text.rfind('.'), truncated_text.rfind('?'), truncated_text.rfind('!'))
        # Ensure punctuation is not the very first character and is reasonably close to the end
        if last_punc > 0 and last_punc > len(truncated_text) * 0.6:
             return truncated_text[:last_punc+1]
        return truncated_text + "..." # Indicate truncation if no clean break
    return text

def contains_specific_name(text, common_names_sample):
    """Basic check if text contains common specific names (case-insensitive)."""
    words = set(re.findall(r'\b[A-Z][a-z]+\b', text)) # Look for capitalized words
    # Also check common names that might not be capitalized if they appear mid-sentence (less likely but possible)
    lower_words = {word.lower() for word in re.findall(r'\b\w+\b', text)}

    potential_names = words.union(lower_words)

    for word in potential_names:
        if word.lower() in common_names_sample:
            # Basic exclusion list for common capitalized words at sentence start or potential false positives
            if word not in ["The", "A", "An", "I", "Is", "Was", "Do", "Does", "Did", "Will", "What", "Where", "When", "Who", "Why", "How", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]:
                return True
    return False


# Sample of common names (lowercase) - Expand if needed
COMMON_NAMES = {
    "john", "mary", "james", "patricia", "robert", "jennifer", "michael", "linda",
    "william", "elizabeth", "david", "susan", "richard", "jessica", "joseph", "sarah",
    "thomas", "karen", "charles", "nancy", "christopher", "lisa", "daniel", "betty",
    "matthew", "margaret", "anthony", "sandra", "mark", "ashley", "donald", "kimberly",
    "steven", "emily", "paul", "donna", "andrew", "michelle", "joshua", "dorothy",
    "kevin", "carol", "brian", "amanda", "george", "melissa", "edward", "deborah",
    "ronald", "stephanie", "timothy", "rebecca", "jason", "sharon", "jeffrey", "laura",
    "ryan", "cynthia", "jacob", "kathleen", "gary", "amy", "nicholas", "shirley",
    "eric", "angela", "jonathan", "helen", "stephen", "anna", "larry", "brenda",
    # Add more common names as needed
    "peter", "samantha", "alex", "chloe", "ben", "zoe"
}


def load_existing_data():
    """Loads existing prompt-response pairs from the file."""
    global existing_pairs, saved_count
    existing_pairs.clear()
    saved_count = 0
    update_status(f"Attempting to load data from '{FILE_NAME}'...")
    try:
        with open(FILE_NAME, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Check structure and types
                    if isinstance(data, dict) and \
                       "prompt" in data and isinstance(data["prompt"], str) and \
                       "response" in data and isinstance(data["response"], str):

                        # Validate prompt format
                        if not data["prompt"].strip().startswith("User:") or "\nAI:" not in data["prompt"]:
                             print(f"Warning [L:{line_num}]: Skipping line - prompt format invalid: {data['prompt'][:50]}...")
                             continue

                        # Validate response format
                        if not data["response"].strip().endswith(END_OF_TEXT_TOKEN):
                            print(f"Warning [L:{line_num}]: Skipping line - response missing end token: {data['response'][-50:]}...")
                            continue

                        # Normalize for duplicate check
                        prompt_norm = ' '.join(data["prompt"].strip().splitlines()).strip()
                        prompt_norm = ' '.join(prompt_norm.split())
                        response_norm = ' '.join(data["response"].strip().split())

                        existing_pairs.add((prompt_norm, response_norm))
                        saved_count += 1
                    else:
                        print(f"Warning [L:{line_num}]: Skipping line with unexpected structure/types: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Warning [L:{line_num}]: Skipping invalid JSON line: {line.strip()}")
                except Exception as e:
                    print(f"Warning [L:{line_num}]: Error processing line: {e} - Line: {line.strip()}")

        update_status(f"Loaded {saved_count} valid existing pairs from '{FILE_NAME}'.")
        update_saved_count_label()
    except FileNotFoundError:
        update_status(f"'{FILE_NAME}' not found. Will create a new one.")
    except Exception as e:
        messagebox.showerror("Load Error", f"Failed to load existing data: {e}")


def update_status(message):
    """Appends a message to the status text area in a thread-safe way."""
    def _update():
        if status_area:
            status_area.config(state=tk.NORMAL)
            status_area.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
            status_area.see(tk.END)
            status_area.config(state=tk.DISABLED)
    if root and root.winfo_exists(): # Check if root window still exists
        root.after(0, _update)


def update_saved_count_label():
    """Updates the label showing the count of saved pairs."""
    def _update():
        if count_label and count_label.winfo_exists(): # Check if label exists
            count_label.config(text=f"Saved Pairs: {saved_count}")
    if root and root.winfo_exists():
        root.after(0, _update)


def call_ollama(prompt_text, generating_user_query=False):
    """Sends prompt to Ollama API. Returns response text or None on error."""
    status_suffix = "(for User Query)" if generating_user_query else "(for AI Response)"
    update_status(f"Sending request to Ollama {status_suffix}...")
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                 "temperature": 0.8, # Keep slightly higher temp for conversational flow
                 "top_p": 0.9,
                 "repeat_penalty": 1.18, # Slightly higher penalty might help reduce simple loops
                 "num_predict": USER_QUERY_NUM_PREDICT if generating_user_query else RESPONSE_NUM_PREDICT,
                 # Stop sequences - critical to prevent unwanted generation
                 "stop": [
                     END_OF_TEXT_TOKEN, "\nUser:", "User:", "\nAI:", "AI:",
                     "</s>", "<|im_end|>", "Human:", "<|user|>", "<|assistant|>",
                     # Add common instruction/task words if they creep in
                     "Explain", "Define", "Summarize", "Translate", "Write a", "Generate a"
                     ]
            }
        }

        response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()

        response_data = response.json()
        generated_text = response_data.get("response", "").strip()

        # Post-processing: Remove potential instruction remnants if stop tokens failed
        lines = generated_text.splitlines()
        cleaned_lines = [line for line in lines if not line.strip().lower().startswith(("user:", "ai:", "human:"))]
        generated_text = "\n".join(cleaned_lines).strip()


        if not generated_text:
             update_status(f"Ollama returned an empty response {status_suffix}.")
             return None

        update_status(f"Received response from Ollama {status_suffix}.")
        return generated_text

    except requests.exceptions.ConnectionError:
        update_status("Error: Could not connect to Ollama API. Is it running?")
        if automation_running: stop_automation() # Stop if connection fails during run
        # Don't show messagebox from background thread directly if possible
        # Rely on status updates and potential manual stop.
        # Consider queueing messagebox for main thread if essential.
        return None
    except requests.exceptions.Timeout:
        update_status("Error: Request to Ollama timed out.")
        return None
    except requests.exceptions.RequestException as e:
        update_status(f"Error: Ollama API request failed: {e}")
        if "404" in str(e) and automation_running:
             # messagebox.showerror("Ollama Error", f"Model '{MODEL_NAME}' not found. Make sure it's pulled in Ollama.")
             update_status(f"CRITICAL ERROR: Model '{MODEL_NAME}' not found. Stopping automation.")
             stop_automation()
        return None
    except Exception as e:
        update_status(f"Error: An unexpected error occurred during Ollama call: {e}")
        return None

def save_pair_to_jsonl(prompt_str, response_str_with_token):
    """Appends a prompt-response pair to the JSONL file in the specified format."""
    global saved_count
    response_final = response_str_with_token.strip()
    if not response_final.endswith(END_OF_TEXT_TOKEN):
        # This shouldn't happen often now, but good fallback
        print(f"Warning: Appending missing {END_OF_TEXT_TOKEN} just before saving.")
        response_final += END_OF_TEXT_TOKEN

    data = {"prompt": prompt_str, "response": response_final}
    try:
        with open(FILE_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
        saved_count += 1
        update_saved_count_label() # Use thread-safe update
        return True
    except Exception as e:
        update_status(f"Error: Failed to save pair: {e}")
        return False


def automation_loop():
    """The main loop for generating and saving User/AI interactions."""
    global automation_running, existing_pairs, session_user_queries

    while automation_running:
        update_status("-" * 20)

        # 1. Generate User Query (Conversational)
        update_status("Attempting to generate a conversational user query...")
        user_query_raw = call_ollama(META_PROMPT, generating_user_query=True)

        if user_query_raw is None:
            if not automation_running: break
            update_status("User query generation failed. Pausing before retry...")
            time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
            continue

        # Normalize, check length, uniqueness, and specific names
        user_query_norm = ' '.join(user_query_raw.strip().split())
        query_word_count = count_words(user_query_norm)

        if not user_query_norm:
             update_status("Generated user query was empty. Skipping.")
             continue

        # Check for specific names
        if contains_specific_name(user_query_norm, COMMON_NAMES):
            update_status(f"Generated user query contains potential name. Skipping: '{user_query_norm[:80]}...'")
            time.sleep(0.5)
            continue

        # Check for non-conversational keywords (basic check)
        non_convo_keywords = ["explain", "define", "summarize", "write a story", "generate code", "what is"]
        if any(keyword in user_query_norm.lower() for keyword in non_convo_keywords):
            update_status(f"Generated user query seems non-conversational. Skipping: '{user_query_norm[:80]}...'")
            time.sleep(0.5)
            continue

        # Enforce max prompt length
        if query_word_count > MAX_PROMPT_WORDS:
            update_status(f"Generated user query too long ({query_word_count} words). Truncating...")
            user_query_norm = truncate_to_words(user_query_norm, MAX_PROMPT_WORDS)
            query_word_count = count_words(user_query_norm)
            if not user_query_norm:
                 update_status("User query empty after truncation. Skipping.")
                 continue

        # Check session uniqueness (case-insensitive)
        session_check_query = user_query_norm.lower()
        if session_check_query in session_user_queries:
            update_status("Generated user query is too similar to one from this session. Skipping.")
            time.sleep(random.uniform(0.5, 1.5))
            continue
        else:
            session_user_queries.add(session_check_query)
            update_status(f"Generated User Query ({query_word_count} words): {user_query_norm[:100]}...")

        # 2. Construct the Full Prompt for the AI
        full_prompt_for_ai = f"User: {user_query_norm}\nAI:"

        # 3. Generate AI Response (Conversational Reply)
        update_status(f"Attempting to generate AI response (max {MAX_RESPONSE_WORDS} words)...")
        ai_response_raw = call_ollama(full_prompt_for_ai, generating_user_query=False)

        if ai_response_raw is None:
            if not automation_running: break
            update_status("AI Response generation failed. Skipping this query.")
            continue

        # Normalize, check length, and specific names
        ai_response_norm = ' '.join(ai_response_raw.strip().split())
        response_word_count = count_words(ai_response_norm)

        if not ai_response_norm:
             update_status("Generated AI response was empty. Skipping.")
             continue

        # Check for specific names in response
        if contains_specific_name(ai_response_norm, COMMON_NAMES):
            update_status(f"Generated AI response contains potential name. Skipping pair.")
            time.sleep(0.5)
            continue

        # Enforce max response length
        if response_word_count > MAX_RESPONSE_WORDS:
            update_status(f"Generated AI response too long ({response_word_count} words). Truncating...")
            ai_response_norm = truncate_to_words(ai_response_norm, MAX_RESPONSE_WORDS)
            response_word_count = count_words(ai_response_norm)
            if not ai_response_norm:
                 update_status("AI response empty after truncation. Skipping.")
                 continue

        # 4. Add the end token
        response_to_save = ai_response_norm + END_OF_TEXT_TOKEN
        update_status(f"Generated AI Response ({response_word_count} words): {ai_response_norm[:100]}...{END_OF_TEXT_TOKEN}")

        # 5. Check for Duplicates (Full Pair) & Save
        prompt_norm_for_check = ' '.join(full_prompt_for_ai.strip().splitlines()).strip()
        prompt_norm_for_check = ' '.join(prompt_norm_for_check.split())
        current_pair = (prompt_norm_for_check, response_to_save)

        if ai_response_norm == "":
             update_status("Skipping: Response became empty after processing.")
        elif current_pair in existing_pairs:
            update_status("Skipping: Exact prompt-response pair already exists in file.")
        else:
            update_status("Unique pair found. Saving...")
            if save_pair_to_jsonl(full_prompt_for_ai, response_to_save):
                existing_pairs.add(current_pair)
                update_status(f"Successfully saved pair #{saved_count}.")
            else:
                update_status("Failed to save the pair to the file.")

        # 6. Delay
        if automation_running:
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            update_status(f"Waiting for {delay:.1f} seconds...")
            time.sleep(delay)

    update_status("Automation stopped.")
    # Update buttons via main thread
    if root and root.winfo_exists():
        root.after(0, lambda: [
            start_button.config(state=tk.NORMAL) if start_button and start_button.winfo_exists() else None,
            stop_button.config(state=tk.DISABLED) if stop_button and stop_button.winfo_exists() else None
        ])


def start_automation():
    """Starts the generation loop in a background thread."""
    global automation_running
    if automation_running:
        return

    # Connection & Model Checks (Keep these)
    update_status("Checking Ollama connection...")
    try:
        base_url = OLLAMA_API_URL.replace("/api/generate", "/")
        if not base_url.endswith('/'): base_url += '/'
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200 and "Ollama is running" in response.text:
             update_status("Ollama connection successful.")
        else:
             details = response.text[:200] if response.text else f"Status: {response.status_code}"
             raise requests.exceptions.RequestException(f"Unexpected response from Ollama base URL ({base_url}): {details}")
    except requests.exceptions.RequestException as e:
        # Use root.after to show messagebox from main thread
        root.after(0, lambda: messagebox.showerror("Ollama Error", f"Cannot connect to or verify Ollama at its base URL.\nPlease ensure it's running and accessible.\nError: {e}"))
        return

    update_status(f"Checking if model '{MODEL_NAME}' is available...")
    try:
        models_response = requests.get(OLLAMA_API_URL.replace("/generate", "/tags"), timeout=15)
        models_response.raise_for_status()
        models_data = models_response.json()
        # More robust check for model name (handles tags like :latest)
        available_models = [m['name'] for m in models_data.get('models', [])]
        if not any(m.startswith(MODEL_NAME + ":") or m == MODEL_NAME for m in available_models):
             root.after(0, lambda: messagebox.showwarning("Model Check", f"Model '{MODEL_NAME}' not found in available models: {available_models}. The script will try to use it anyway, but it might fail."))
             update_status(f"Warning: Model '{MODEL_NAME}' not listed by Ollama. Proceeding anyway.")
        else:
             update_status(f"Model '{MODEL_NAME}' appears to be available.")
    except requests.exceptions.RequestException as e:
         update_status(f"Warning: Could not verify available models: {e}. Proceeding anyway.")
    except Exception as e:
         update_status(f"Warning: Error parsing model list: {e}. Proceeding anyway.")


    automation_running = True
    session_user_queries.clear()
    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    update_status("Automation started (Daily Conversations Focus)...") # << UPDATED Status

    thread = threading.Thread(target=automation_loop, daemon=True)
    thread.start()

def stop_automation():
    """Signals the generation loop to stop."""
    global automation_running
    if not automation_running:
        return
    update_status("Stop requested. Finishing current cycle (if any)...")
    automation_running = False
    if stop_button and stop_button.winfo_exists(): # Check if widget exists
        stop_button.config(state=tk.DISABLED)

# --- Setup GUI (Mostly unchanged, updated title) ---
root = tk.Tk()
# Updated title
root.title(f"Ollama Daily Conversation Generator ({MODEL_NAME})") # << UPDATED Title
root.geometry("700x550")

# Control Frame
control_frame = tk.Frame(root)
control_frame.pack(pady=10)

start_button = tk.Button(control_frame, text="Start Automation", command=start_automation, width=15)
start_button.pack(side=tk.LEFT, padx=10)

stop_button = tk.Button(control_frame, text="Stop Automation", command=stop_automation, width=15, state=tk.DISABLED)
stop_button.pack(side=tk.LEFT, padx=10)

count_label = tk.Label(control_frame, text="Saved Pairs: 0", width=15)
count_label.pack(side=tk.LEFT, padx=10)

# Status Area
status_frame = tk.Frame(root)
status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

tk.Label(status_frame, text="Status Log:").pack(anchor="w")
status_area = scrolledtext.ScrolledText(status_frame, height=25, width=80, wrap=tk.WORD, state=tk.DISABLED)
status_area.pack(fill=tk.BOTH, expand=True)

# Assign widgets AFTER creation
status_area_ref = status_area
count_label_ref = count_label
start_button_ref = start_button
stop_button_ref = stop_button

# Load data and Run
root.after(100, load_existing_data)

# Ensure automation stops if window is closed
def on_closing():
    global automation_running
    if automation_running:
        print("Window closed, stopping automation...")
        automation_running = False
    # Wait briefly for thread to potentially notice the flag
    # This isn't guaranteed, but might help slightly. Daemon threads will terminate anyway.
    # time.sleep(0.1)
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
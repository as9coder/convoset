import json
import requests
import threading
import time
import random
import re
import signal # To handle Ctrl+C gracefully

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate" # Default Ollama API endpoint
MODEL_NAME = "mistral"  # Or whichever model you have pulled in Ollama
FILE_NAME = "daily_conversations_data_console_2.jsonl" # New filename for console version
END_OF_TEXT_TOKEN = "<|endoftext|>"

# --- Generation Constraints ---
MAX_PROMPT_WORDS = 100
MAX_RESPONSE_WORDS = 200
RESPONSE_NUM_PREDICT = int(MAX_RESPONSE_WORDS * 1.6) # Give buffer
USER_QUERY_NUM_PREDICT = int(MAX_PROMPT_WORDS * 1.6) # Buffer for user query too

# --- META_PROMPT (Conversation Focus) ---
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

# --- Globals ---
existing_pairs = set()
session_user_queries = set()
automation_running = False
saved_count = 0
automation_thread = None # To hold the thread object

# --- Functions ---

def log_status(message):
    """Prints a message to the console with a timestamp."""
    print(f"{time.strftime('%H:%M:%S')} - {message}")

def count_words(text):
    """Counts words in a string, handling basic punctuation."""
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def truncate_to_words(text, max_words):
    """Truncates text to a maximum number of words."""
    words = text.split()
    if len(words) > max_words:
        truncated_text = ' '.join(words[:max_words])
        last_punc = max(truncated_text.rfind('.'), truncated_text.rfind('?'), truncated_text.rfind('!'))
        if last_punc > 0 and last_punc > len(truncated_text) * 0.6:
             return truncated_text[:last_punc+1]
        return truncated_text + "..."
    return text

def contains_specific_name(text, common_names_sample):
    """Basic check if text contains common specific names (case-insensitive)."""
    words = set(re.findall(r'\b[A-Z][a-z]+\b', text))
    lower_words = {word.lower() for word in re.findall(r'\b\w+\b', text)}
    potential_names = words.union(lower_words)
    for word in potential_names:
        if word.lower() in common_names_sample:
            if word not in ["The", "A", "An", "I", "Is", "Was", "Do", "Does", "Did", "Will", "What", "Where", "When", "Who", "Why", "How", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]:
                return True
    return False

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
    "peter", "samantha", "alex", "chloe", "ben", "zoe"
}

def load_existing_data():
    """Loads existing prompt-response pairs from the file."""
    global existing_pairs, saved_count
    existing_pairs.clear()
    saved_count = 0
    log_status(f"Attempting to load data from '{FILE_NAME}'...")
    try:
        with open(FILE_NAME, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict) and \
                       "prompt" in data and isinstance(data["prompt"], str) and \
                       "response" in data and isinstance(data["response"], str):
                        if not data["prompt"].strip().startswith("User:") or "\nAI:" not in data["prompt"]:
                             print(f"   WARNING [L:{line_num}]: Skipping line - prompt format invalid: {data['prompt'][:50]}...")
                             continue
                        if not data["response"].strip().endswith(END_OF_TEXT_TOKEN):
                            print(f"   WARNING [L:{line_num}]: Skipping line - response missing end token: {data['response'][-50:]}...")
                            continue
                        prompt_norm = ' '.join(data["prompt"].strip().splitlines()).strip()
                        prompt_norm = ' '.join(prompt_norm.split())
                        response_norm = ' '.join(data["response"].strip().split())
                        existing_pairs.add((prompt_norm, response_norm))
                        saved_count += 1
                    else:
                        print(f"   WARNING [L:{line_num}]: Skipping line with unexpected structure/types: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"   WARNING [L:{line_num}]: Skipping invalid JSON line: {line.strip()}")
                except Exception as e:
                    print(f"   WARNING [L:{line_num}]: Error processing line: {e} - Line: {line.strip()}")
        log_status(f"Loaded {saved_count} valid existing pairs from '{FILE_NAME}'.")
    except FileNotFoundError:
        log_status(f"'{FILE_NAME}' not found. Will create a new one.")
    except Exception as e:
        log_status(f"ERROR: Failed to load existing data: {e}")
        # Consider exiting if loading fails critically? For now, just log.

def call_ollama(prompt_text, generating_user_query=False):
    """Sends prompt to Ollama API. Returns response text or None on error."""
    status_suffix = "(for User Query)" if generating_user_query else "(for AI Response)"
    log_status(f"Sending request to Ollama {status_suffix}...")
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                 "temperature": 0.8,
                 "top_p": 0.9,
                 "repeat_penalty": 1.18,
                 "num_predict": USER_QUERY_NUM_PREDICT if generating_user_query else RESPONSE_NUM_PREDICT,
                 "stop": [
                     END_OF_TEXT_TOKEN, "\nUser:", "User:", "\nAI:", "AI:",
                     "</s>", "<|im_end|>", "Human:", "<|user|>", "<|assistant|>",
                     "Explain", "Define", "Summarize", "Translate", "Write a", "Generate a"
                     ]
            }
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        response_data = response.json()
        generated_text = response_data.get("response", "").strip()
        lines = generated_text.splitlines()
        cleaned_lines = [line for line in lines if not line.strip().lower().startswith(("user:", "ai:", "human:"))]
        generated_text = "\n".join(cleaned_lines).strip()
        if not generated_text:
             log_status(f"Ollama returned an empty response {status_suffix}.")
             return None
        log_status(f"Received response from Ollama {status_suffix}.")
        return generated_text
    except requests.exceptions.ConnectionError:
        log_status("ERROR: Could not connect to Ollama API. Is it running?")
        if automation_running: stop_automation() # Signal stop
        return None
    except requests.exceptions.Timeout:
        log_status("ERROR: Request to Ollama timed out.")
        return None
    except requests.exceptions.RequestException as e:
        log_status(f"ERROR: Ollama API request failed: {e}")
        if "404" in str(e) and automation_running:
             log_status(f"CRITICAL ERROR: Model '{MODEL_NAME}' not found. Stopping automation.")
             stop_automation() # Signal stop
        return None
    except Exception as e:
        log_status(f"ERROR: An unexpected error occurred during Ollama call: {e}")
        return None

def save_pair_to_jsonl(prompt_str, response_str_with_token):
    """Appends a prompt-response pair to the JSONL file and logs count."""
    global saved_count
    response_final = response_str_with_token.strip()
    if not response_final.endswith(END_OF_TEXT_TOKEN):
        print(f"   WARNING: Appending missing {END_OF_TEXT_TOKEN} just before saving.")
        response_final += END_OF_TEXT_TOKEN
    data = {"prompt": prompt_str, "response": response_final}
    try:
        with open(FILE_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
        saved_count += 1
        log_status(f"Successfully saved pair #{saved_count}.") # Log count here
        return True
    except Exception as e:
        log_status(f"ERROR: Failed to save pair: {e}")
        return False

def automation_loop():
    """The main loop for generating and saving User/AI interactions."""
    global automation_running, existing_pairs, session_user_queries

    while automation_running:
        log_status("-" * 20)

        # 1. Generate User Query (Conversational)
        log_status("Attempting to generate a conversational user query...")
        user_query_raw = call_ollama(META_PROMPT, generating_user_query=True)
        if user_query_raw is None:
            if not automation_running: break # Check if stopped during Ollama call
            log_status("User query generation failed. Pausing before retry...")
            time.sleep(random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
            continue

        # --- Processing User Query ---
        user_query_norm = ' '.join(user_query_raw.strip().split())
        query_word_count = count_words(user_query_norm)
        if not user_query_norm:
             log_status("Generated user query was empty. Skipping.")
             continue
        if contains_specific_name(user_query_norm, COMMON_NAMES):
            log_status(f"Generated user query contains potential name. Skipping: '{user_query_norm[:80]}...'")
            time.sleep(0.5); continue
        non_convo_keywords = ["explain", "define", "summarize", "write a story", "generate code", "what is", "list"]
        if any(keyword in user_query_norm.lower() for keyword in non_convo_keywords):
            log_status(f"Generated user query seems non-conversational. Skipping: '{user_query_norm[:80]}...'")
            time.sleep(0.5); continue
        if query_word_count > MAX_PROMPT_WORDS:
            log_status(f"Generated user query too long ({query_word_count} words). Truncating...")
            user_query_norm = truncate_to_words(user_query_norm, MAX_PROMPT_WORDS)
            query_word_count = count_words(user_query_norm)
            if not user_query_norm:
                 log_status("User query empty after truncation. Skipping."); continue
        session_check_query = user_query_norm.lower()
        if session_check_query in session_user_queries:
            log_status("Generated user query is too similar to one from this session. Skipping.")
            time.sleep(random.uniform(0.5, 1.5)); continue
        else:
            session_user_queries.add(session_check_query)
            log_status(f"Generated User Query ({query_word_count} words): {user_query_norm[:100]}...")
        # --- End User Query Processing ---

        # 2. Construct Prompt for AI
        full_prompt_for_ai = f"User: {user_query_norm}\nAI:"

        # 3. Generate AI Response
        log_status(f"Attempting to generate AI response (max {MAX_RESPONSE_WORDS} words)...")
        ai_response_raw = call_ollama(full_prompt_for_ai, generating_user_query=False)
        if ai_response_raw is None:
            if not automation_running: break
            log_status("AI Response generation failed. Skipping this query."); continue

        # --- Processing AI Response ---
        ai_response_norm = ' '.join(ai_response_raw.strip().split())
        response_word_count = count_words(ai_response_norm)
        if not ai_response_norm:
             log_status("Generated AI response was empty. Skipping."); continue
        if contains_specific_name(ai_response_norm, COMMON_NAMES):
            log_status(f"Generated AI response contains potential name. Skipping pair."); time.sleep(0.5); continue
        if response_word_count > MAX_RESPONSE_WORDS:
            log_status(f"Generated AI response too long ({response_word_count} words). Truncating...")
            ai_response_norm = truncate_to_words(ai_response_norm, MAX_RESPONSE_WORDS)
            response_word_count = count_words(ai_response_norm)
            if not ai_response_norm:
                 log_status("AI response empty after truncation. Skipping."); continue
        response_to_save = ai_response_norm + END_OF_TEXT_TOKEN
        log_status(f"Generated AI Response ({response_word_count} words): {ai_response_norm[:100]}...{END_OF_TEXT_TOKEN}")
        # --- End AI Response Processing ---

        # 4. Check Duplicates & Save
        prompt_norm_for_check = ' '.join(full_prompt_for_ai.strip().splitlines()).strip()
        prompt_norm_for_check = ' '.join(prompt_norm_for_check.split())
        current_pair = (prompt_norm_for_check, response_to_save)
        if ai_response_norm == "":
             log_status("Skipping: Response became empty after processing.")
        elif current_pair in existing_pairs:
            log_status("Skipping: Exact prompt-response pair already exists in file.")
        else:
            log_status("Unique pair found. Saving...")
            if save_pair_to_jsonl(full_prompt_for_ai, response_to_save):
                existing_pairs.add(current_pair)
                # Count is logged within save_pair_to_jsonl now
            else:
                log_status("Failed to save the pair to the file.") # Error already logged

        # 5. Delay
        if automation_running:
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            log_status(f"Waiting for {delay:.1f} seconds...")
            time.sleep(delay)

    log_status("Automation loop finished.")


def check_ollama_connection():
    """Checks connection to Ollama base URL and model availability."""
    log_status("Checking Ollama connection...")
    try:
        base_url = OLLAMA_API_URL.replace("/api/generate", "/")
        if not base_url.endswith('/'): base_url += '/'
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200 and "Ollama is running" in response.text:
             log_status("Ollama connection successful.")
        else:
             details = response.text[:200] if response.text else f"Status: {response.status_code}"
             raise requests.exceptions.RequestException(f"Unexpected response from Ollama base URL ({base_url}): {details}")
    except requests.exceptions.RequestException as e:
        log_status(f"ERROR: Cannot connect to or verify Ollama at its base URL: {e}")
        log_status("Please ensure Ollama is running and accessible.")
        return False # Indicate connection failure

    log_status(f"Checking if model '{MODEL_NAME}' is available...")
    try:
        models_response = requests.get(OLLAMA_API_URL.replace("/generate", "/tags"), timeout=15)
        models_response.raise_for_status()
        models_data = models_response.json()
        available_models = [m['name'] for m in models_data.get('models', [])]
        if not any(m.startswith(MODEL_NAME + ":") or m == MODEL_NAME for m in available_models):
             log_status(f"WARNING: Model '{MODEL_NAME}' not found in available models: {available_models}.")
             log_status("The script will try to use it anyway, but it might fail.")
        else:
             log_status(f"Model '{MODEL_NAME}' appears to be available.")
    except requests.exceptions.RequestException as e:
         log_status(f"WARNING: Could not verify available models: {e}. Proceeding anyway.")
    except Exception as e:
         log_status(f"WARNING: Error parsing model list: {e}. Proceeding anyway.")

    return True # Indicate checks passed (or warnings were issued but proceeding)

def start_automation():
    """Sets the flag and starts the automation thread."""
    global automation_running, automation_thread
    if automation_running:
        log_status("Automation is already running.")
        return

    log_status("Starting automation (Daily Conversations Focus)...")
    session_user_queries.clear()
    automation_running = True
    # Make thread non-daemon so we can join it cleanly on exit
    automation_thread = threading.Thread(target=automation_loop, daemon=False)
    automation_thread.start()

def stop_automation():
    """Signals the generation loop to stop."""
    global automation_running
    if not automation_running:
        return
    log_status("Stop requested. Finishing current cycle (if any)...")
    automation_running = False # Signal the loop

def signal_handler(sig, frame):
    """Handles Ctrl+C interruption."""
    print() # Print newline after ^C
    log_status("Ctrl+C detected. Stopping automation gracefully...")
    stop_automation()

# --- Main Execution ---
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler) # Register Ctrl+C handler

    load_existing_data()

    if not check_ollama_connection():
        log_status("Exiting due to Ollama connection issues.")
        exit(1) # Exit with an error code

    start_automation()

    # Keep the main thread alive to listen for signals and wait for the worker thread
    while automation_running:
        try:
            # Check thread health periodically (optional)
            if automation_thread and not automation_thread.is_alive():
                log_status("Worker thread unexpectedly stopped.")
                automation_running = False # Ensure loop exits
                break
            time.sleep(1) # Wait politely
        except InterruptedError: # Can happen during sleep on signal
            log_status("Main thread interrupted during sleep.")
            stop_automation() # Ensure stop is signalled again if needed
            break

    # Wait for the automation thread to finish its current cycle after being signalled
    if automation_thread and automation_thread.is_alive():
        log_status("Waiting for automation thread to complete...")
        automation_thread.join() # Wait for the thread to finish

    log_status("Program finished.")

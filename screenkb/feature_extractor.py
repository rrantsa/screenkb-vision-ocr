"""Feature Extractor — converts OCR + Layout into a feature list."""

from __future__ import annotations

import re
from typing import List, Set

from .ocr_engine import OCRResult
from .layout_analyzer import LayoutResult

# ---------------------------------------------------------------------------
# Keyword → feature mappings
# Each entry: (feature_name, list_of_trigger_words_or_patterns)
# ---------------------------------------------------------------------------
_KEYWORD_FEATURES: List[tuple[str, List[str]]] = [
    # Errors / tracebacks
    ("traceback", ["Traceback", "traceback", "most recent call"]),
    ("python_error", ["Python", "SyntaxError", "NameError", "ValueError", "TypeError",
                      "AttributeError", "ImportError", "ModuleNotFoundError", "KeyError",
                      "IndexError", "RuntimeError", "Exception", "raise"]),
    ("javascript_error", ["TypeError:", "ReferenceError:", "SyntaxError:", "Uncaught",
                          "console.error", "at Object.", "at Function.", "undefined is not"]),
    ("stack_trace", ["at ", "line ", "File \""]),
    ("http_error", ["404", "403", "401", "500", "502", "503", "Bad Gateway",
                    "Not Found", "Forbidden", "Unauthorized", "Internal Server Error"]),

    # Shell / terminal
    ("terminal", ["$", "#", "bash", "zsh", "cmd", "powershell", "C:\\", "PS >",
                  "root@", "user@", ">>>"]),
    ("git", ["git ", "commit", "branch", "merge", "push", "pull", "diff",
             "HEAD", "origin", "remote"]),

    # IDE / code editor
    ("editor", ["def ", "function ", "class ", "import ", "require(", "const ",
                "let ", "var ", "return ", "if (", "for ("]),
    ("sidebar", []),   # injected from layout
    ("toolbar", []),   # injected from layout

    # Web / browser
    ("browser", ["http://", "https://", "www.", ".com", ".org", ".net",
                 "localhost", "127.0.0.1"]),
    ("login_form", ["Login", "Sign in", "Username", "Password",
                    "Connexion", "Mot de passe", "Se connecter",
                    "Adresse email", "mot de passe oublié", "continuer avec",
                    "Forgot password", "Remember me"]),
    ("form", ["Submit", "Cancel", "Save", "Input", "Select", "Checkbox",
              "Required", "Enregistrer", "Annuler"]),
    ("modal", ["Close", "×", "Dialog", "Modal", "Confirm", "Are you sure"]),
    ("table", ["thead", "tbody", "th>", "td>", "Row", "Column", "Sort"]),
    ("navigation", ["Menu", "Home", "Settings", "Dashboard", "Profile",
                    "Logout", "Sign out", "Déconnexion"]),
    ("notification", ["Success", "Error", "Warning", "Info", "Alert",
                      "Succès", "Erreur", "Avertissement"]),

    # Specific apps
    ("vscode", ["Explorer", "Extensions", "Source Control", "Run and Debug",
                "Problems", "Output", "Debug Console", "TERMINAL"]),
    ("chrome_browser", ["New Tab", "Reload", "Bookmark", "chrome://", "DevTools"]),
    ("excel_spreadsheet", ["xlsx", "xls", "Sheet1", "Sheet2", "Cell", "Formula"]),
    ("pdf_viewer", [".pdf", "Page", "Zoom", "Rotate", "Acrobat", "Reader"]),

    # QA / testing specific
    ("test_runner", ["PASS", "FAIL", "SKIP", "passed", "failed", "skipped",
                     "assertions", "describe(", "it(", "test(", "expect("]),
    ("api_response", ["200 OK", "201 Created", "application/json",
                      "\"status\":", "\"message\":", "\"data\":", "\"error\":",
                      "Content-Type:", "curl ", "Postman"]),
    ("network_log", ["GET ", "POST ", "PUT ", "DELETE ", "PATCH ",
                     "Request", "Response", "Headers", "Payload"]),
]

# Application name detection
_APP_KEYWORDS: List[tuple[str, List[str]]] = [
    ("VS Code", ["Visual Studio Code", "vscode", ".vscode", "Explorer", "Extensions",
                 "Source Control", "Debug Console", "TERMINAL", "PROBLEMS"]),
    ("Chrome", ["Google Chrome", "chrome://", "Chrome DevTools", "New Tab – Google Chrome"]),
    ("Firefox", ["Mozilla Firefox", "firefox", "about:blank"]),
    ("Terminal", ["bash", "zsh", "PowerShell", "cmd.exe", "Terminal", "iTerm"]),
    ("Slack", ["Slack", "slack.com", "workspace", "channel", "direct message"]),
    ("Jira", ["Jira", "atlassian.net", "Issue", "Sprint", "Backlog", "Epic"]),
    ("Linear", ["Linear", "linear.app", "Issue", "Cycles", "Projects", "Triage"]),
    ("Figma", ["Figma", "figma.com", "Frame", "Component", "Prototype"]),
    ("Notion", ["Notion", "notion.so", "notion.site"]),
    ("GitHub", ["GitHub", "github.com", "Pull request", "Issues", "Commits", "Actions"]),
    ("Excel", ["Microsoft Excel", ".xlsx", "Sheet", "Workbook"]),
    ("Word", ["Microsoft Word", ".docx", "Document"]),
    ("Postman", ["Postman", "Collection", "Environment", "Send", "Authorization", "Body"]),
    ("Docker", ["Docker", "container", "image", "Dockerfile", "docker-compose"]),
]


def extract_features(ocr: OCRResult, layout: LayoutResult) -> List[str]:
    """Build a deduplicated feature list from OCR text and detected layout regions."""
    features: Set[str] = set()
    text_lower = (ocr.raw_text or "").lower()
    words_lower = {w.lower() for w in (ocr.words or [])}

    # Inject layout-based features
    for label in (layout.region_labels or []):
        features.add(label)

    # Match keyword rules against full text and word set
    for feature_name, triggers in _KEYWORD_FEATURES:
        if not triggers:
            continue  # pure layout features — already injected above
        for trigger in triggers:
            t_lower = trigger.lower()
            if t_lower in text_lower or t_lower in words_lower:
                features.add(feature_name)
                break

    return sorted(features)


def detect_application(ocr: OCRResult) -> str:
    """Guess the application name from OCR text."""
    text_lower = (ocr.raw_text or "").lower()
    for app_name, keywords in _APP_KEYWORDS:
        for kw in keywords:
            if kw.lower() in text_lower:
                return app_name
    return "Unknown"

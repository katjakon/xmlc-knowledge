
PREFIX_PROMPT = "Titel: "
SUFFIX_PROMPT = ". Schlagwörter: "
INSTRUCT_PROMPT = "Du bist ein hilfreicher Assistent für Bibliothekare. Gib mir einige Schlagwörter für diesen Buchtitel: {}. Schlagwörter: "
SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent für Bibliothekare. Du hilfst bei der Katalogisierung von Büchern. Du gibst nur die Schlagwörter zurück, die für den Titel relevant sind. Du gibst keine Erklärungen oder Kommentare ab. Du gibst ausschließlich die Schlagwörter zurück, sonst nichts. "
SYSTEM_PROMPT_SHORT = "Du bist ein hilfreicher Assistent für Bibliothekare. Du hilfst bei der Katalogisierung von Büchern. "
USER_PROMPT = "Gib mir einige Schlagwörter für diesen Buchtitel: {}. Schlagwörter: " 
CONTEXT_PROMPT = """Ähnliche oder verwandte Schlagwörter könnten sein: {}."""
FS_PROMPT = "Titel: {}. Schlagwörter: {}"
EXAMPLE = "Beispiel: Titel: Die Entdeckung der Langsamkeit. Schlagwörter: Entdeckung; Langsamkeit; Natur; Wissenschaft; Philosophie. "
SYSTEM_PROMPT_EXAMPLE = SYSTEM_PROMPT_SHORT + EXAMPLE
INFO = "Relevante Information:\n"
RELATION_MAPPING = {
    "broader": " ist eine Art von ",
    "related": " ist verwandt mit "
}

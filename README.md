# World Cup RAG System — LangChain Agent + Gemini + FAISS

An end-to-end Retrieval-Augmented Generation system for querying 145 years of FIFA World Cup history. A LangChain ReAct agent routes each user query through a six-tool pipeline — semantic retrieval, head-to-head computation, LLM synthesis, and prediction reporting — before returning a grounded, evidence-backed answer.

---

## How it works

User queries are embedded with `all-MiniLM-L6-v2` and matched against a FAISS index of ~900 World Cup match records. A ReAct agent then decides which combination of tools to invoke. Some queries go straight to the retrieval tool; others chain through the reasoning tool first, then the LLM synthesis tool, then the prediction report generator. Conversation memory (LangChain `ConversationBufferMemory`) keeps multi-turn context coherent within a session.

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            ReAct Agent  ·  Gemini 2.5 Flash (temp=0.3)      │
│                  LangChain ConversationBufferMemory          │
└──┬──────────┬──────────┬──────────┬──────────┬─────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌─────────┐ ┌─────────┐ ┌───────────────┐
│data  │ │data    │ │retrieval│ │reasoning│ │ llm_synthesis │
│disco-│ │ingest- │ │_tool    │ │_tool    │ │ _tool         │
│very  │ │ion     │ │         │ │         │ │               │
└──────┘ └────────┘ └────┬────┘ └────┬────┘ └───────┬───────┘
                         │           │               │
                         ▼           ▼               ▼
                    ┌─────────┐ ┌─────────┐   ┌───────────┐
                    │  FAISS  │ │  Match  │   │  Gemini   │
                    │  Index  │ │  Data-  │   │  2.5 Flash│
                    │ ~900 doc│ │  frame  │   └─────┬─────┘
                    └─────────┘ └────┬────┘         │
                                     │         ┌────▼──────────────┐
                                     └────────►│report_generation  │
                                               │_tool              │
                                               │(chains reasoning  │
                                               │+ Gemini)          │
                                               └───────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                         Answer                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Tools

| Tool | What it does |
|------|-------------|
| `dataset_discovery_tool` | Returns dataset schema, date range, and available teams |
| `data_ingestion_tool` | Summary stats: match counts, goal averages, most frequent teams |
| `retrieval_tool` | Semantic similarity search over FAISS index — used for historical match questions |
| `reasoning_tool` | Exact H2H computation: wins, losses, draws, goal tallies, last-5 form |
| `llm_synthesis_tool` | Passes retrieved context to Gemini for natural language synthesis |
| `report_generation_tool` | Chains reasoning + Gemini to produce structured prediction reports |

The agent prompt includes explicit routing hints (`Use retrieval_tool for history, reasoning_tool for H2H, report_generation_tool for predictions`) to reduce unnecessary tool calls.

---

## Dataset

**Source:** [Kaggle — International Football Results 1872–2017](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) (martj42)

The dataset is filtered to FIFA World Cup main tournament matches only — qualification rounds are excluded. Each match is converted to a natural-language sentence for embedding:

```
On July 13, 2014, in the FIFA World Cup, Germany played against Argentina
in Rio de Janeiro, Brazil. The score was Germany 1 - 0 Argentina. Result: home win.
```

**Data cutoff: 2017.** Post-2018 World Cup matches (Russia 2018, Qatar 2022) are not in the dataset. Prediction queries are grounded in historical records only — the agent does not have knowledge of current squads, injuries, or recent form beyond 2017.

---

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemini 2.5 Flash (`langchain-google-genai`) |
| Agent | LangChain ReAct (`create_react_agent`) |
| Vector store | FAISS (`langchain-community`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Data | Pandas |
| Visualization | Plotly |
| UI | Streamlit |
| Deployment | localtunnel (Colab) |

---

## Setup & Run

**1. Install dependencies**
```bash
pip install "langchain>=0.3,<1.0" "langchain-core>=0.3,<1.0" "langchain-community>=0.3,<1.0" \
    langchain-google-genai faiss-cpu sentence-transformers pandas numpy plotly
```

**2. Set your Gemini API key**

In Colab (recommended): add `GEMINI_API_KEY` to Colab Secrets.

Or directly:
```python
os.environ["GOOGLE_API_KEY"] = "your-key-here"
```

**3. Run the notebook**
```bash
jupyter notebook worldcup_chatbot_Code.ipynb
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
# or in Colab:
!npx localtunnel --port 8501 &
!streamlit run app.py --server.port 8501 &
```

The FAISS index is built on first run and cached to `cache/faiss_index/` — subsequent runs load from cache.

---

## Example queries

- *"What happened in the 2014 World Cup final?"* — routes through `retrieval_tool`
- *"Compare Brazil and Germany's World Cup record"* — routes through `reasoning_tool`
- *"Predict a World Cup match between Argentina and France"* — chains `reasoning_tool` → `report_generation_tool`
- *"What World Cup data do you have?"* — routes through `dataset_discovery_tool`

---

## Design notes & limitations

- **`reasoning_tool` uses exact string matching** to identify team names from the query. Typos or alternate names (e.g., "West Germany" vs "Germany") may cause it to fail to find both teams and return an error.
- **`ConversationBufferMemory` has no token cap.** Long sessions accumulate unbounded history, which will eventually hit Gemini's context window limit.
- **`llm_synthesis_tool` is a pass-through** — it wraps any string in a Gemini prompt. The agent can route queries there directly without retrieval, which means not all answers are grounded in the vector DB.
- **No post-2017 data.** Any question about 2018, 2022, or 2026 World Cup events will fall outside the dataset.

---

## Author

**Arun V** — [github.com/v-arun2002](https://github.com/v-arun2002)

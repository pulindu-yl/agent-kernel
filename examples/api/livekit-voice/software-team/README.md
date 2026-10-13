# Agent Kernel Realtime Software Team

> **Showcase example.** This is a full application built on Agent Kernel, not a minimal recipe. To learn the handoffs it is built on, start with the small [`google/handoffs`](../google/handoffs/) example.

A LiveKit voice call between you, the client, and an AI software team. Three voice agents (Worker, PM and Architect) talk to you and to each other. They agree the requirements and a design with you, challenge each other's ideas, and ask you only business questions. Two background agents (Coder and Reviewer) then build the software in a sandbox, test it and review it, and the voice team reports back into the call. You follow it all on one screen: the call on the left, and on the right a live project board with what the team has agreed, what it is building, and what it is waiting for you to decide. At the end you get a folder with the requirements, the design, the code and the test results.

Every agent is a **Google ADK** agent. The voice agents run on Gemini Live, each in its own voice. The background agents run on a text model through ADK's `LiteLlm`: OpenAI's `gpt-4.1` by default, or any other LiteLLM model.

## The team

| Member | Kind | Job | Tools |
|---|---|---|---|
| You | Client | Business answers; approve the requirements and the design, with a click on your page | — |
| `pm` | Voice, entry agent | Opens the meeting, business questions, requirements and acceptance criteria | `start_project`, `record_requirement`, `revise_requirement`, `drop_requirement`, `record_assumption`, `record_question`, `confirm_requirements`, `check_acceptance`, `settle_assumption` |
| `worker` | Voice | Plans tasks, drives development and delivery | `create_tasks`, `start_development`, `deliver` |
| `architect` | Voice | Leads the design debate and the code review, approves the code | `record_decision`, `record_component`, `write_design`, `request_review`, `rule_on_finding`, `approve_code`, `settle_assumption` |
| `coder` | Background, text model | Writes the code and tests in the sandbox, and notes the choices it made | sandbox tools, `report_progress`, `ask_team`, `record_assumption`, `record_build_note` |
| `reviewer` | Background, text model | Reads the real code, runs the tests, finds issues, checks acceptance | sandbox tools, `submit_review`, `submit_acceptance` |

Every voice agent also has `get_project_state`, `get_progress` and `answer_coder` (the Architect answers the Coder's technical questions after hearing the Worker's view, the PM its business ones).

## Where to start reading

```
software-team/
├── server.py            START HERE: builds the team and serves it in the LiveKit room
├── members/             WHO is on the team: the agent definitions
│   ├── prompts.py         what each agent is told, in plain English
│   ├── voice.py           Worker, PM and Architect (Gemini Live), and how they hand over
│   └── background.py      Coder and Reviewer (a text model through LiteLLM)
├── team/                HOW the team works together: plain Python, no LLM
│   ├── tools.py           every agent's tools
│   ├── project.py         what the team agreed and did (the shared state)
│   ├── documents.py       the markdown documents written from it
│   ├── models.py          the records: requirement, decision, task, finding, ...
│   ├── rules.py           the process order, and the limit on handoffs
│   ├── jobs.py            runs the Coder and the Reviewer, then the tests
│   ├── call_link.py       speaks into the live call at quiet moments
│   ├── progress.py        turns background activity into entries on the board
│   ├── state.py           one call's state, found from the session a tool runs in
│   ├── board.py           the project board: one more participant that shows the client the project
│   ├── room_access.py     LiveKit tokens for the board and the client's page
│   ├── approval.py        your approval of the requirements and the design, from a button on your page
│   ├── client_questions.py  notices when an agent has asked you something, from its words
│   ├── log_watch.py       reading a line another library logs: the example's shortcut for three events
│   ├── project_files.py   the project's files for your page: view, download, share
│   └── client_page.py     serves the client's page and its token on port 8000
├── panel/               the client's screen: a small React app (the call + the project board)
├── config.yaml          realtime mode, and the sandbox for the background agents
└── demo_test.py         tests for the coordination code: no model, room or Docker needed
```

Read them in this order:

1. **`server.py`**: `SoftwareTeamServer.load()` registers the voice team and the background agents with Agent Kernel, and `run()` serves the PM, the entry agent, in the LiveKit room.
2. **`members/voice.py`**: `VoiceTeam.build()` defines the three voice agents as ordinary ADK agents. The PM opens the meeting, as a project manager does with a client, so the call does not begin with a handover. The Worker and the Architect are the PM's `sub_agents`, so any of the three can hand the call to another with ADK's `transfer_to_agent`.
3. **`members/prompts.py`**: each voice agent's instruction is `MEETING_RULES` plus its role. The rules are where the behaviour lives: how to hand over, when to ask the client, how to report progress.
4. **`members/background.py`**: `BackgroundTeam` defines the Coder and the Reviewer, ADK agents too, on `LiteLlm(model="openai/gpt-4.1")`, as in the `examples/api/adk` example.
5. **`team/tools.py`**: what the agents can do. A voice agent's job tool (`start_development`, `request_review`, ...) queues a job in `team/jobs.py` and returns at once.

## How it works

**Voice is the meeting room, not the engine.** A realtime voice model is built for conversation: it writes weak code, and a long tool call would hold up its turn. So the voice agents only talk, hand over to each other, record what was agreed, and start work with short tools. The real work runs in the background:

```
VOICE AGENT    ITS TOOL               BACKGROUND AGENT   THAT AGENT'S TOOLS
worker    -->  start_development  --> coder         -->  write_sandbox_file, run_command (pytest), report_progress, ask_team
architect -->  request_review     --> reviewer      -->  read_sandbox_file, run_command (pytest), submit_review
pm        -->  check_acceptance   --> reviewer      -->  the same, plus submit_acceptance
```

A job tool returns at once with a job id, so the voice agent keeps talking. The job runs on its own thread, never on the realtime pool's event loop, so it cannot stall the call's audio; so does the rewriting of the project's documents after each change. When the job finishes, its result comes back into the call as a `[Team update, not from the client]` message, and whichever agent has the call tells you about it and takes the next step.

What keeps it honest and moving:

- **Shared state.** `ProjectStore` holds the requirements (with acceptance criteria such as `R1.2`), assumptions, open questions, decisions with the options they rejected, components, tasks, progress, review findings and acceptance results. Every voice and background tool reads and writes it, and the background agents get it as their brief.
- **The order holds.** The prompts ask for the SDLC order, and `ProcessGates` makes it hold. The client approves the requirements and the design together, so approval waits for a written design (which itself needs a recorded decision with the challenge behind it), and tasks and development wait for the approval. A review waits for a build. A finding is fixed only once the Architect has ruled it should be, after asking the Worker about it. Acceptance waits for the Architect's approval of the latest code, which waits for every finding to be confirmed fixed by the Reviewer (not just by the Coder) or waived with a reason. Delivery waits for an acceptance check made after the last build, covering every current requirement, with every criterion passing, and for every assumption the Coder made alone to be settled; a requirement added later needs a new check. Every check runs again when a queued job starts, so a delivery requested during a new build does not ship it unchecked. A tool asked to skip ahead answers "Not ..." with what is missing and who should act, and the agent hands over.
- **Agreement is earned.** The rules forbid opening with praise and allow agreement only once an objection has been answered. `record_decision` takes the challenge raised and how it was answered, and the design cannot be written without it.
- **Evidence, not claims.** After every development job, `DevJobRunner` runs `pytest` in the sandbox itself and records the real output; the voice agents are told never to claim results a tool did not report. A review or an acceptance check counts only once the Reviewer submits it: one that finishes without `submit_review` or `submit_acceptance` is asked once more, then fails, rather than passing as a clean review. If the tests fail at delivery, or no code could be copied out of the sandbox, the project is not marked delivered.
- **One sandbox workspace.** The Coder and the Reviewer run in the call's own background AK session (`room_01-dev`), so they share one sandbox and take turns. They never use the call's session, whose lock every voice tool call takes. Each run starts from its brief (`include_contents="none"`): the brief carries the project, so a run does not resend every earlier job's transcript to the model.
- **Only the background agents get the sandbox.** `sandbox.agents: [coder, reviewer]` in `config.yaml`; the voice agents never see `run_command`.
- **Few handoffs, complete turns.** Every handoff opens a new Gemini Live session, sent the agent's instructions, its tools and the conversation so far: 5,000 to 9,000 tokens each, so a quick back-and-forth went over the free tier's 65,000 input tokens a minute. So each agent says everything it has in one turn (up to five sentences), and outside the design debate the team hands over only when the next step needs someone else. There is no limit on how often the team hands over, only checks in a `before_tool_callback` on `transfer_to_agent`, which answers in the tool's place:
  - the agent that starts a job keeps the call until its result comes; while a job runs, a handover goes through only if something new came since that agent got the call (you spoke or typed, a team update, a question from the Coder). Live, while a review ran, the call bounced between the Worker and the PM, each repeating the status;
  - one transfer per turn, and in the design debate or the code review none from an agent that has said nothing since it got the call: live, a double transfer and silent bounces counted as rounds of the debate;
  - `HandoffGuard` refuses a 7th handoff in a row with nothing recorded, no job finished and nothing from you; the agent then tells you where the team has got to, if it has not just done so. A finished job resets the count, so the handover its result needs (to the Architect, to approve the code) always goes through. A refusal is never mentioned to you.
- **A real design debate.** The Architect and the Worker argue the design out over two to four rounds before anything is built, because agreement in one polite exchange ("sounds good, we'll do it like that") is neither realistic nor interesting to listen to. The prompts set the tone (sparring partners, concrete cases, no praise); the code makes it hold: `write_design` waits for at least two rounds, a fifth is refused (the Architect then settles what is left), and the handover brief tells the Architect which round it is in. The Worker hears only that the Architect is waiting for its answer (told the round, it once acted out every round alone in one turn), and each later round has to raise a new point, so no round is the same answer again. A design changed after it was written is debated again.
- **Every handover says why.** An agent handing over tells the teammate, by name, what it wants: a question, or the step that is now theirs. The transfer's result (an `after_tool_callback`) also tells the agent taking the call where the project stands and who acts next, so an agent handed the call mid-design gives its view instead of saying "I can't build yet" and handing it straight back. It is worded as facts, not orders: the new agent's socket is seeded the way ADK seeds one, and ADK presents every earlier agent turn and tool result, the transfer's included, as quoted data, never as instructions.
- **Updates at quiet moments.** Text sent into a call starts a model response, and on Gemini Live it cuts off one still being generated, so `CallLink` waits until nothing is being played, you are not speaking, and the model has answered everything it was last given: your words, anything you typed, a tool's result (an agent just handed the call takes a moment to start), and the link's own last update. It reads that from the connection's turn state (`is_responding`, `is_caller_speaking`, `last_model_output_at`, `last_caller_speech_at`, `last_edge_text_at`, `last_tool_result_at`). On Gemini, a long answer of yours keeps `last_caller_speech_at` current only as far as the model sends transcriptions while you speak. An update that no longer applies when the call is quiet (a Coder question that timed out) is dropped instead.
- **The client sees the project, not just hears it.** `ProjectBoard` joins the room as one more participant, "Project board", with no microphone. Whenever the project changes, it sends a snapshot of it (at most once a second) as JSON on its own topic, `ak.project`, to everyone in the room except the agents, and it sends the current one to every participant that joins, so a page opened or reloaded mid-call starts full. The client's page (`panel/`) draws it beside the call. It never reaches the agents: Agent Kernel's LiveKit gateway takes only audio and the chat topics as the client's input, and the board sends neither. No Agent Kernel change was needed.
- **Corrections change the record.** When you correct a requirement ("I said cosmetics, not cost metrics"), the PM changes it with `revise_requirement` or removes it with `drop_requirement`, and you confirm the requirements again; its old acceptance results no longer count, and a dropped requirement's id is never reused. You can point at the board ("R1.1 should say three items").
- **A dropped model connection does not start the project over.** A new Gemini Live connection starts at the PM and remembers nothing of the conversation, but the project is still recorded: `CallLink` notices the new connection and tells the PM where the project stands, and your name, and `start_project` refuses to rename a project whose requirements are confirmed. Only the latest renewal's notice is posted, written when it is posted: live, failed reconnect attempts each queued one, and the Worker, told thirteen times that it "delivers the project", delivered thirteen times. A job already queued or running is not queued again, and a delivered project is delivered again only after a new build.
- **A relaxed meeting.** The agents talk like a calm team, not a call centre: they never rush you, ask you to wait or stay longer, apologize for how long something takes, or talk about their own rules and limits. A step the process does not allow yet is sorted out between them, without telling you. While a job runs and nobody needs an answer, they simply stop talking until there is news.
- **Progress you can see.** A long job never looks like a broken call.
  - When a job starts, the agent says in one sentence what the team has started. The job itself waits until you have heard that: the model writes its words faster than they are spoken and a tool runs the moment it is called, so a review (a few seconds) used to be done on the board while the Architect was still saying it would ask for one.
  - Every tool call the Coder and the Reviewer make is recorded as it happens (ADK's `after_tool_callback`). So is the model making them wait for a rate limit, from the `openai` client's retry log.
  - The board's running-job card shows the latest of it ("coder wrote app/main.py", "coder ran the tests: 4 passed"), with how long the job has run. Ask "How's it going?" and the agent answers from `get_progress`.
  - Running work is not narrated into the call. Every update posted into a live call makes the agent answer it, and a voice model cannot stay silent, so narration every 45 s came out as the same status line again and again. Only what needs the team posts: a job finishing or failing, the Coder's question or assumption, your click or file on the page.
  - The same activity is in `log.md` and in the server log (`software_team.activity`).
- **No stalls.** A realtime model only speaks when something starts its turn, and a teammate hears nothing until the call is handed to it with the transfer tool. Voice models tend to announce a handover ("I'll bring in the Architect") instead of making it, so the prompts say to call the tool in the same turn and never to announce it. Once you have spoken or typed, `CallLink` nudges the team if nobody speaks for 5 seconds while no job is running, no question is waiting for you and the plan is not waiting for your approval. The nudge says to continue without repeating itself, and to call the transfer tool if the agent only named a teammate. It nudges at most twice until you speak or type.
- **Each thing said once.** An agent asks you a question just by saying it, with no tool around it: a tool answering in the middle of a spoken turn makes Gemini Live start talking again, and the PM once asked the same question three times. For the same reason the job tools' replies state facts ("Started development job J3") and never ask the agent to say something, which made an agent that had already announced the job announce it again. After a handover the agent taking the call reads the conversation so far, with every agent's words marked with its name, its own included; the rules tell it that lines with its own name are what it already said, and that only the latest team update is current, so it does not redo an earlier one's "Next:" step. `create_tasks` adds no task twice. The team still knows you owe an answer: `ClientQuestionWatcher` reads each agent turn the gateway logs once it has played, and a turn that ends by asking you something (not addressing a teammate, not handing over) holds back the nudges and shows the question under **Waiting for you**; an answer you type counts too. The page also drops tool-call text Gemini sometimes transcribes (`call:transfer_to_agent{...}`), even mid-sentence.
- **No build loops.** A build that ends with failing tests names them in its update, and the next build gives the Coder the end of the failing output, so a rebuild fixes what failed instead of starting over (live, the Worker had started 13 identical builds, ten seconds apart, none of them told what failed). When 3 builds in a row still end with the same failing tests, `start_development` is refused until something changes: the Architect records a different approach from the failures `get_progress` lists, the PM drops or simplifies that part with you, or you speak or type. A note such as "asked the client" does not count as a change.
- **No retry loops.** A failed background run (a retired model, the sandbox down) fails its job, and the voice team is told not to restart it. After two failures in a row, that kind of job is refused until you speak or type, so you can say "try again" once the problem is fixed.
- **A real code review.** The background Reviewer reads the real code and lists the problems; the review itself is a conversation, at most two rounds. The Architect puts every finding to the Worker as a question about their code ("Why does the booking check run before the insert?"). The Worker speaks for the build: it agrees a fix, or defends the code with a reason from the Coder's build notes, the design decisions or the requirements, and never invents one. The Architect then rules on each finding with `rule_on_finding`: fix it, or waive it with the reason. Only findings ruled to fix are built; the Reviewer re-checks the fixes; then the Architect approves the code (`approve_code`), and only then does the PM check acceptance and tell you how the review went. `ProcessGates` makes it hold: no ruling before the Worker was asked, no fix without a ruling, a third round refused, no acceptance without the approval of the latest code. The board's **Code review** card shows each finding's ruling and reason, and `status.md` keeps them.
- **Build notes.** The Coder records each choice the design did not settle, with the reason (`record_build_note`: "prices are stored in cents: no rounding errors"). They are in `design.md` and in what the voice agents read, so "why is it done this way?" has a real answer.
- **Questions from the Coder.** `ask_team` reaches the call as a team update. A technical question is talked through: the Worker brings it to the Architect with its view, and the Architect decides and answers with `answer_coder`. The PM answers a business one, and asks you only if you have not decided it yet. Deciding out loud does not reach the Coder. With no answer in two minutes, the Coder records an assumption and carries on, and a question still waiting to be posted is dropped. Every assumption the Coder records is posted to the call and must be settled (`settle_assumption`) before delivery: the PM checks a business rule with you, the Architect decides a technical one.
- **Honest task status.** A task is done only when the Coder reports it done and the tests pass; passing tests alone do not show that every task was built. If the Coder finishes without reporting any task, the job asks it once more which it finished. The Coder's and the Reviewer's own words are understood ("completed" is done, "passed" is a pass); anything else is sent back. Unfinished tasks are named in the job's update, the Worker starts development again for them, and delivery waits until every task is done. A review can resolve only a finding the Coder fixed: an open one waits for the Architect's ruling.

## Quickstart

This example uses Agent Kernel features that are newer than the latest release (ADK tool callbacks, turn state and authored history in realtime). Until they are released, build Agent Kernel from this repository and install it locally:

```bash
(cd ../../../../ak-py && ./build.sh)
./build.sh local
```

Once released, `./build.sh` is enough.

### 1. Set up

```bash
export GEMINI_API_KEY="..."
export AK_LIVEKIT__URL="wss://your-project.livekit.cloud"
export AK_LIVEKIT__API_KEY="..."
export AK_LIVEKIT__API_SECRET="..."
```

The Coder and the Reviewer make many model calls per job, more than the Gemini API free tier allows, so they run on OpenAI by default:

```bash
export OPENAI_API_KEY="..."
```

The sandbox uses Docker by default: start the Docker daemon first. Without Docker, `export AK_SANDBOX__DEFAULT_PROFILE=local` runs the generated code on this machine in a temp folder, with **no isolation**; use it for local development only.

| Variable | Default | What it sets |
|---|---|---|
| `TEAM_TEXT_MODEL` | `openai/gpt-4.1` | The background agents' model, as a LiteLLM `provider/model` (for example `anthropic/claude-sonnet-5-5`, with that provider's key) |
| `TEAM_OUTPUT_DIR` | `./output` | Where each project's folder is written |

### 2. Build the client's page (once)

```bash
cd panel && npm install && npm run build && cd ..
```

It needs Node.js 22.18 or newer. To change the page while the server runs, use `npm run dev` instead and open http://localhost:5173.

### 3. Run the server

```bash
uv run server.py
```

Open **http://localhost:8000**, enter your name and press **Join the call**, then speak. You join under your name, so the PM greets you by it ("Hi Pulindu") and the team uses it. The page shows:

- **left, the call:** a tile for you, the PM, the Architect and the Worker, each with its own avatar; the speaker's tile glows with their voice, and the agent that has the call is marked "On the call". Below, the conversation: what each agent said, and what you said, as Gemini heard it (so a mishearing shows at once). Agent Kernel logs each of your turns (`Caller said: ...`); the example's `ClientWordsWatcher` passes that line to the board, which sends it to your page. No Agent Kernel change: it is the same line you see in the terminal.
- **below the conversation:** a message box for anything easier written than said (a link, a name, an exact figure), which the gateway passes to the agent with the call as your words, and a 📎 button to **share a file**: it is saved in the project's `shared/` folder, announced to the team (a short text file is quoted to them, as data), listed in `requirements.md`, and put in the sandbox under `shared/` for the Coder. The voice agents cannot open a picture or a PDF; the Coder can use it.
- **bottom:** the microphone (mute, and a menu to choose another microphone), the call's length, and Leave. If your default microphone cannot start (another tab holds it, or a Bluetooth headset in music mode has none), the page tries each microphone in turn.
- **right, the project board:** the phase and the steps (Requirements, Design, Approval, Build, Review, Acceptance, Delivered), then the tabs. **Project:** **Ready for your approval**, once the design is written: the requirements and the design to scroll through, with **Looks good, build it** and **I'd like a change** (the page brings it into view once); **Waiting for you** (a question the team asked, or an assumption to confirm), the running job's latest activity, the requirements with a ✓ or ✗ per acceptance criterion once checked, the design decisions with the challenge behind each, the tasks and test results, the **Code review** (each finding to discuss, to fix, fixed or waived, with the Architect's reason, and the Architect's approval), and the delivered folder. **Documents:** the requirements, the design and the status as the team writes them, updated live, with **Save as PDF** (your browser's print, of the document alone). **Files:** every file of the project, to view in place or download, and **Download the project (.zip)**, which the Delivered card offers too. **Activity:** the team's log, newest first.

The page gets its LiveKit token from `GET /token`, reads and shares the project's files through `GET /project/files`, `GET /project/files/{path}`, `GET /project/download` and `POST /project/share`, and sends your approval through `POST /project/approval`, all on the same server. None of them asks who is calling, so the server listens on this machine only (`api.host: 127.0.0.1` in `config.yaml`). To open it to your network, set `AK_API__HOST=0.0.0.0`, and put authentication in front of it first: anyone who reaches it can join the call, read the project, approve the plan and share files with the team.

You can still join from the LiveKit Sandbox dashboard instead (room `room_01`); you then hear the team but see no board. Every voice agent joins the room as its own participant (`agent-pm`, `agent-worker`, `agent-architect`), and the board as `project-board`.

### 4. Try it: "I want a login system"

1. *"Hi team, I need a login system for my web app."* The PM starts the project.
2. The PM asks you business questions, one at a time, such as whether users sign up themselves and whether they need password reset. Answer them; say *"you decide"* to let the team assume something.
3. The Architect and the Worker debate the design without you, over two to four rounds. The Architect proposes the whole design and attacks it with the concrete ways it could fail (a guessable reset link, two people booking the same slot); the Worker answers every concern and attacks back where the design is more than the requirements need, slow to build or hard to test. Neither agrees because something sounds fine: they give way only to a concrete argument, and say what convinced them. Each decision is recorded with the challenge behind it, so `design.md` shows the debate and how many rounds it took.
4. Once the design is written, the board shows **Ready for your approval**: what the team will build and how, to scroll through. Press **Looks good, build it** (or just say so), or **I'd like a change** and tell the PM what. A click is recorded at once and the team hears it; the Worker then creates the tasks and starts development.
5. Ask *"How's it going?"* while the Coder works. The Worker answers from the recorded progress, and the board shows the Coder's latest steps.
6. When the build is done, the Architect requests a review and holds a code review with the Worker: "why is it done this way?", the Worker's answer, and a ruling on each finding (fix, or waive with the reason). The fixes are built and re-checked, and the Architect approves the code. The PM then checks every acceptance criterion against the tests and tells you how the review went.
7. The Worker delivers, and tells you where the project is.

You speak about six times, all about the business. The whole run takes roughly ten minutes.

## What you get

```
output/login-system/
├── requirements.md     requirements, acceptance criteria, assumptions, open questions
├── design.md           components, and decisions with their reasons and rejected options
├── status.md           tasks, background jobs, review findings, acceptance results and the outcome
├── log.md              who recorded what, and when
├── test-results.txt    the last raw pytest output
└── code/               the app and its tests, copied out of the sandbox on delivery
```

The documents are rewritten on every change, so you can watch them fill in during the call. A project never reuses a folder an earlier call left: a second "login system" goes to `output/login-system-2/`.

## Tests

`demo_test.py` covers the coordination code with fakes: no model, LiveKit room or Docker needed.

```bash
uv run pytest demo_test.py
```

## Limits

- **One more LiveKit connection.** The project board is a participant for the whole call, like each agent, and LiveKit bills connection minutes for it too.
- **No preview of the built app.** The board shows what was built and tested, not the running app: the sandbox cannot expose a port yet.
- **Audio buffer.** `config.yaml` keeps 300 ms of the agents' audio ahead of playback (`execution.realtime.playback_lead_ms`; the default is 50 ms). This server also runs the background agents and the sandbox, and with less the audio broke up. Lower it for a snappier start if your machine keeps up.
- **One call at a time per process, in one process.** `CallLink` and the jobs run in-process, so the example runs on the `in_memory` transport only, with one Agent Runner.
- **Use a Gemini API key with billing enabled.** Every handoff opens a new Gemini Live session (5,000 to 9,000 tokens), and the free tier allows 65,000 input tokens a minute for Gemini Live. A quick design discussion can go over it, and the call then fails with a quota error (1011). `HandoffGuard` stops handover loops and the design debate is capped at four rounds, but nothing limits how often the team hands over.
- **About ten minutes.** Gemini Live session limits are not handled, and the client is expected to stay on the call until delivery. Project state lasts only as long as the process.
- **Python projects only.** The Coder writes Python with pytest tests: a web API, or a website served by FastAPI with HTML pages. The voice agents know it and offer that when a client asks for another stack (React, Next.js, Node). The sandbox itself could run Node with an image that has it, but the team's test evidence (`DevJobRunner` runs and reads pytest) and the Coder's instructions are Python-only; other stacks, and previewing a built site, are follow-ups.
- **A short gap on every handoff** (about 1 to 2 seconds while the new agent's socket opens), as in the `google/handoffs` example.
- **Prompts drive the meeting.** How well the agents follow the meeting rules depends on the model; expect to tune `members/prompts.py`.
- **No pull request.** Delivery writes a folder; opening a pull request, merging and deploying are out of scope.

---
name: arena
description: Compare a few independently developed solutions through adversarial review, revision, and blind judging using Codex subagents. Use when the user asks for Arena or competing approaches, especially when concrete tests can resolve disagreements.
---

# Arena for Codex

Adapted from Jakeschincariol/arena-skill (MIT; see LICENSE). The Python state machine, strategy cards, five scoring criteria, and prompt templates are retained. This local copy defaults to four competitors and waves of two. It is stored here at the user's request; it is not globally installed or automatically discovered outside this chat.

## Run

1. Freeze the user's standalone task and evidence constraints in a task file. Specify writable directories, prohibited side effects, current baseline, resource limits, deadline and what validation actually establishes. Do not turn an aspiration into a promised outcome.
2. Use `python3 <skill-dir>/bracket.py plan --agents N --wave W`, then `init --agents N --wave W --seed S --task-file ABS_PATH --dir ABS_RUN`. Keep the working directory and ABS_RUN inside the user's authorized output root. Never reuse another agent's `.arena/LATEST`.
3. Drive `next --dir ABS_RUN`, `prompts PHASE --dir ABS_RUN`, `check PHASE --dir ABS_RUN`, `collect --dir ABS_RUN`, and `advance --dir ABS_RUN`. The briefs are generated from [templates.md](templates.md). Scoring follows [rubric.md](rubric.md).
4. Replace Claude Agent calls with `collaboration.spawn_agent(task_name=...,fork_turns="none",message="Read <absolute brief path> and follow it exactly. It is your whole brief.")`. Do not override model settings unless authorized. Run at most the available child slots, normally two. After each wave completes, launch the next. A follow-up may reuse a competitor's agent for its attack/defense, but a judge must be independent of the solutions being judged and never receive strategy cards or the orchestrator's preferred outcome. Do not let children spawn further agents.
5. Competitors may execute bounded tests under their scratch paths when the task authorizes it. Prevent output conflicts and simultaneous heavy jobs through disjoint scratch paths and a shared resource lock specified in the task. Judges verify cited artifacts instead of rewarding confident prose.
6. The main agent coordinates but does not compete or choose winners. Read receipts/state during the tournament, then read the champion only after `winner`. Do not report the champion as empirically best unless the task's validation supports that statement. A valid outcome is that none qualifies for adoption.
7. Missing outputs: retry once; record NO OUTPUT only after a second failure. A missing judge receives a fresh independent retry. Stop at the user's stop/cutoff and report completed versus pending work. Preserve state for resumption.

## Deliver

Return the winner's concrete result, evidence and limits, material surviving attacks, run path, and actual competitor/round/call counts. Honor the user's existing authorization for implementation; do not introduce a new approval requirement. The state machine itself does not execute subagents or apply project changes.

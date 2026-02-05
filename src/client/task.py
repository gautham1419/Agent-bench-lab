import enum
import json
import requests
import re

from src.typings import *
from src.utils import *
from .agent import AgentClient


class TaskError(enum.Enum):
    START_FAILED = "START_FAILED"
    INTERACT_FAILED = "INTERACT_FAILED"
    AGENT_FAILED = "AGENT_FAILED"
    NETWORK_ERROR = "NETWORK_ERROR"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class TaskClient:
    def __init__(
        self, name: str, controller_address: str = "http://localhost:5000/api", *_, **__,
    ) -> None:
        self.name = name
        self.controller_address = controller_address
        print("TaskClient created: {} ({})".format(name, controller_address))

    def get_indices(self) -> List[SampleIndex]:
        result = requests.get(
            self.controller_address + "/get_indices", params={"name": self.name}
        )
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        return result.json()

    def get_concurrency(self) -> int:
        try:
            result = requests.get(
                self.controller_address + "/list_workers"
            )
        except Exception as e:
            print(ColorMessage.yellow(f"Warning task {self.name} cannot connect to controller {e}"))
            return 0
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        result = result.json()
        if self.name not in result:
            print(ColorMessage.yellow(f"task {self.name} not found in worker list"))
            return 0
        concurrency = 0
        for worker in result[self.name]["workers"].values():
            
            status = worker.get("status")
            is_alive = (status == WorkerStatus.ALIVE) or (
                isinstance(status, str) and status.strip().upper() == "ALIVE"
            )
            if is_alive:
                concurrency += worker["capacity"] - worker["current"]
                    
        return concurrency
    
    def run_sample(self, index: SampleIndex, agent: AgentClient) -> TaskClientOutput:
        try:
            start_resp = requests.post(
                self.controller_address + "/start_sample",
                json=StartSampleRequest(name=self.name, index=index).dict(),
            )
        except Exception as e:
            return TaskClientOutput(error=TaskError.NETWORK_ERROR.value, info=str(e))

        if start_resp.status_code == 406:
            return TaskClientOutput(error=TaskError.NOT_AVAILABLE.value, info=start_resp.text)

        if start_resp.status_code != 200:
            return TaskClientOutput(error=TaskError.START_FAILED.value, info=start_resp.text)

        try:
            result = start_resp.json()
        except Exception as e:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value,
                info=f"Invalid JSON from start_sample: {e}\n{start_resp.text}",
            )

        sid_header = start_resp.headers.get("Session_id") or start_resp.headers.get("Session-Id")
        if not sid_header:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value,
                info=f"Missing Session_id header. headers={dict(start_resp.headers)} body={result}",
            )

        try:
            sid = int(sid_header)
        except Exception:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value,
                info=f"Invalid Session_id header value: {sid_header}",
            )

        latest_result = result

        def _cancel():
            try:
                requests.post(
                    self.controller_address + "/cancel",
                    headers={"Session_id": str(sid), "Session-Id": str(sid)},
                    json={"session_id": sid},
                    timeout=30,
                )
            except Exception:
                pass

        def _is_session_not_found(text: str) -> bool:
            try:
                return "session not found" in (text or "").lower()
            except Exception:
                return False

        conversation_messages = []
        _seen_messages = set()

        def _add_messages(msgs):
            if not msgs:
                return
            for m in msgs:
                try:
                    key = json.dumps(m, sort_keys=True, ensure_ascii=False)
                except Exception:
                    key = str(m)
                if key in _seen_messages:
                    continue
                _seen_messages.add(key)
                conversation_messages.append(m)

        if isinstance(result, dict):
            _add_messages(result.get("messages", []))

        fc_tools = []
        tool_name_set = set()

        def _update_tools(res_dict):
            nonlocal fc_tools, tool_name_set
            if not isinstance(res_dict, dict):
                return
            tools = res_dict.get("tools")
            if tools is None:
                return
            if not isinstance(tools, list):
                return
            fc_tools = tools
            tool_name_set = set()
            for t in fc_tools:
                fn = (t or {}).get("function") or {}
                n = fn.get("name")
                if n:
                    tool_name_set.add(n)

        _update_tools(result)

        def _build_output_history():
            hist = []
            for m in conversation_messages:
                role = m.get("role")
                content = m.get("content")
                if not isinstance(content, str):
                    continue
                if role == "assistant":
                    hist.append(ChatHistoryItem(role="agent", content=content))
                elif role == "user":
                    hist.append(ChatHistoryItem(role="user", content=content))
            return hist

        def _maybe_finish(res_dict):
            if not isinstance(res_dict, dict):
                return None

            status = res_dict.get("status")
            if status and status != "running":
                final_result = dict(res_dict)
                final_result["messages"] = conversation_messages
                final_history = _build_output_history()
                try:
                    return TaskClientOutput(
                        output=TaskOutput(
                            index=index,
                            status=SampleStatus(status),
                            result=final_result,
                            history=final_history,
                        )
                    )
                except Exception:
                    return TaskClientOutput(
                        output=TaskOutput(
                            index=index,
                            status=SampleStatus.UNKNOWN,
                            result=final_result,
                            history=final_history,
                        )
                    )

            if res_dict.get("finish") is True:
                final_result = dict(res_dict)
                final_result["messages"] = conversation_messages
                final_history = _build_output_history()
                return TaskClientOutput(
                    output=TaskOutput(
                        index=index,
                        status=SampleStatus.COMPLETED,
                        result=final_result,
                        history=final_history,
                    )
                )

            if "output" in res_dict:
                return TaskClientOutput(output=res_dict["output"])

            return None

        auto_commit_next = False

        if isinstance(result, dict) and "output" in result:
            while SampleStatus(result["output"]["status"]) == SampleStatus.RUNNING:
                try:
                    content = agent.inference(result["output"]["history"])
                    response = AgentOutput(content=content)
                except AgentContextLimitException:
                    response = AgentOutput(status=AgentOutputStatus.AGENT_CONTEXT_LIMIT)
                except Exception as e:
                    _cancel()
                    return TaskClientOutput(
                        error=TaskError.AGENT_FAILED.value,
                        info=str(e),
                        output=latest_result.get("output") if isinstance(latest_result, dict) else None,
                    )

                try:
                    interact_resp = requests.post(
                        self.controller_address + "/interact",
                        headers={"Session_id": str(sid), "Session-Id": str(sid)},
                        json=InteractRequest(session_id=sid, agent_response=response).dict(),
                    )
                except Exception as e:
                    return TaskClientOutput(
                        error=TaskError.NETWORK_ERROR.value,
                        info=str(e),
                        output=latest_result.get("output") if isinstance(latest_result, dict) else None,
                    )

                if interact_resp.status_code != 200:
                    if not _is_session_not_found(interact_resp.text):
                        _cancel()
                    return TaskClientOutput(
                        error=TaskError.INTERACT_FAILED.value,
                        info=interact_resp.text,
                        output=latest_result.get("output") if isinstance(latest_result, dict) else None,
                    )

                result = interact_resp.json()
                latest_result = result

            return TaskClientOutput(output=result["output"])

        decoder = json.JSONDecoder()

        def _first_json_obj(s: str):
            if not isinstance(s, str):
                return None
            s = s.strip()

            try:
                return json.loads(s)
            except Exception:
                pass

            for i, ch in enumerate(s):
                if ch != "{":
                    continue
                try:
                    obj, _ = decoder.raw_decode(s[i:])
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj
            return None

        def _normalize_tool_payload(obj_dict):
            if not isinstance(obj_dict, dict):
                return None

            name = obj_dict.get("name")
            arguments = obj_dict.get("arguments")
            thought = obj_dict.get("thought", "")

            if name is None and isinstance(obj_dict.get("tool"), str):
                name = obj_dict.get("tool")

            if arguments is None:
                arguments = {}

                if name == "execute_sql":
                    if isinstance(obj_dict.get("query"), str):
                        arguments["query"] = obj_dict.get("query")
                    elif isinstance(obj_dict.get("sql"), str):
                        arguments["query"] = obj_dict.get("sql")
                    elif isinstance(obj_dict.get("answer"), str):
                        arguments["query"] = obj_dict.get("answer")

                if name == "commit_final_answer":
                    if "answers" in obj_dict:
                        arguments["answers"] = obj_dict.get("answers")
                    elif "answer" in obj_dict:
                        a = obj_dict.get("answer")
                        arguments["answers"] = a if isinstance(a, list) else [str(a)]

            if not isinstance(name, str) or not name:
                return None

            if "|" in name and tool_name_set:
                parts = [p.strip() for p in name.split("|") if p.strip()]
                for p in parts:
                    if p in tool_name_set:
                        name = p
                        break

            if not isinstance(arguments, dict):
                arguments = {}

            if name == "execute_sql" and "query" not in arguments and "sql" in arguments:
                arguments["query"] = arguments.pop("sql")

            if tool_name_set and name not in tool_name_set:
                if "execute_sql" in tool_name_set and isinstance(arguments.get("query"), str):
                    name = "execute_sql"
                elif "commit_final_answer" in tool_name_set:
                    name = "commit_final_answer"

            if name == "commit_final_answer" and "answers" not in arguments:
                arguments["answers"] = [""]

            should_auto_commit = False
            if name == "execute_sql":
                q = arguments.get("query", "")
                if isinstance(q, str):
                    q_up = q.lstrip().upper()
                    if q_up.startswith(
                        ("UPDATE", "INSERT", "DELETE", "CREATE", "DROP", "ALTER", "REPLACE", "TRUNCATE")
                    ):
                        should_auto_commit = True

            return name, arguments, thought, should_auto_commit

        max_turns = 200
        for _ in range(max_turns):
            if auto_commit_next and ("commit_final_answer" in tool_name_set):
                auto_commit_next = False
                msg = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "commit_final_answer",
                                "arguments": json.dumps({"answers": [""]}),
                            },
                        }
                    ],
                }

                _add_messages([msg])

                try:
                    interact_resp = requests.post(
                        self.controller_address + "/interact",
                        headers={"Session_id": str(sid), "Session-Id": str(sid)},
                        json={"session_id": sid, "messages": [msg]},
                    )
                except Exception as e:
                    _cancel()
                    return TaskClientOutput(error=TaskError.NETWORK_ERROR.value, info=str(e), output=None)

                if interact_resp.status_code != 200:
                    if not _is_session_not_found(interact_resp.text):
                        _cancel()
                    out = _maybe_finish(latest_result)
                    if out is not None:
                        return out
                    return TaskClientOutput(error=TaskError.INTERACT_FAILED.value, info=interact_resp.text, output=None)

                result = interact_resp.json()
                latest_result = result

                if isinstance(result, dict):
                    _add_messages(result.get("messages", []))
                _update_tools(result)

                out = _maybe_finish(result)
                if out is not None:
                    return out
                continue

            tools = fc_tools

            tool_names = []
            tool_specs_lines = []
            for t in tools:
                fn = (t or {}).get("function") or {}
                n = fn.get("name")
                if not n:
                    continue
                tool_names.append(n)
                params = fn.get("parameters", {})
                try:
                    params_str = json.dumps(params, ensure_ascii=False)
                except Exception:
                    params_str = str(params)
                tool_specs_lines.append(f"{n}: {params_str}")

            if tool_names:
                fc_instruction = (
                    "You must respond ONLY with valid JSON.\n"
                    "Schema:\n"
                    f"{{\"name\":\"{'|'.join(tool_names)}\",\"arguments\":{{}},\"thought\":\"\"}}\n"
                    "Rules:\n"
                    "- Call exactly one tool per turn.\n"
                    "- Put any natural language in `thought` only.\n"
                    "- `arguments` must be a JSON object matching the selected tool's JSON schema.\n"
                    "Tool schemas:\n"
                    + "\n".join(tool_specs_lines)
                )
            else:
                fc_instruction = (
                    "You must respond ONLY with valid JSON.\n"
                    "Schema:\n"
                    "{\"name\":\"bash_action|answer_action|finish_action\",\"arguments\":{},\"thought\":\"\"}\n"
                    "Do not output any extra text outside JSON."
                )

            history = [{"role": "user", "content": fc_instruction}]

            for m in conversation_messages:
                role = m.get("role")
                content = m.get("content")
                if content is None:
                    continue
                if role == "assistant":
                    history.append({"role": "agent", "content": content})
                else:
                    if isinstance(content, str) and content.startswith("Internal error:"):
                        continue
                    history.append({"role": "user", "content": content})

            try:
                content = agent.inference(history)
            except AgentContextLimitException:
                _cancel()
                return TaskClientOutput(error=TaskError.AGENT_FAILED.value, info="Agent context limit", output=None)
            except Exception as e:
                _cancel()
                return TaskClientOutput(error=TaskError.AGENT_FAILED.value, info=str(e), output=None)

            tool_calls = None
            assistant_content = content

            def _try_parse_to_tool_calls(text: str):
                obj = _first_json_obj(text)
                norm = _normalize_tool_payload(obj)
                if norm is None:
                    return None, None, False
                name, arguments, thought, should_auto_commit = norm
                tc = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
                ac = thought if isinstance(thought, str) else ""
                return tc, ac, should_auto_commit

            try:
                tc, ac, should_auto_commit = _try_parse_to_tool_calls(content)
                if tc is not None:
                    tool_calls = tc
                    assistant_content = ac
                    if should_auto_commit:
                        auto_commit_next = True
            except Exception:
                pass

            if tool_calls is None:
                try:
                    retry_history = list(history)
                    retry_history.append(
                        {"role": "user", "content": "Invalid. Output ONLY one JSON object matching schema exactly."}
                    )
                    retry_content = agent.inference(retry_history)
                    tc, ac, should_auto_commit = _try_parse_to_tool_calls(retry_content)
                    if tc is not None:
                        tool_calls = tc
                        assistant_content = ac
                        if should_auto_commit:
                            auto_commit_next = True
                except Exception:
                    pass

            if tool_calls is None and ("commit_final_answer" in tool_name_set):
                tool_calls = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "commit_final_answer",
                            "arguments": json.dumps({"answers": [""]}),
                        },
                    }
                ]
                assistant_content = ""

            msg = {"role": "assistant", "content": assistant_content}
            if tool_calls is not None:
                msg["tool_calls"] = tool_calls

            _add_messages([msg])

            try:
                interact_resp = requests.post(
                    self.controller_address + "/interact",
                    headers={"Session_id": str(sid), "Session-Id": str(sid)},
                    json={"session_id": sid, "messages": [msg]},
                )
            except Exception as e:
                _cancel()
                return TaskClientOutput(error=TaskError.NETWORK_ERROR.value, info=str(e), output=None)

            if interact_resp.status_code != 200:
                if not _is_session_not_found(interact_resp.text):
                    _cancel()
                out = _maybe_finish(latest_result)
                if out is not None:
                    return out
                return TaskClientOutput(error=TaskError.INTERACT_FAILED.value, info=interact_resp.text, output=None)

            result = interact_resp.json()
            latest_result = result

            if isinstance(result, dict):
                _add_messages(result.get("messages", []))
            _update_tools(result)

            out = _maybe_finish(result)
            if out is not None:
                return out

        _cancel()
        return TaskClientOutput(
            error=TaskError.INTERACT_FAILED.value,
            info=f"Exceeded max_turns={max_turns}. Last response: {latest_result}. Collected_messages={len(conversation_messages)}",
            output=None,
        )
    def calculate_overall(self, results: List[TaskOutput]) -> JSONSerializable:
        statistics = {s: 0 for s in SampleStatus}
        for result in results:
            statistics[SampleStatus(result.status)] += 1
        for s in SampleStatus:
            statistics[s] /= len(results)
        statistics["average_history_length"] = sum(
            [len(result.history) for result in results]
        ) / len(results)
        statistics["max_history_length"] = max(
            [len(result.history) for result in results]
        )
        statistics["min_history_length"] = min(
            [len(result.history) for result in results]
        )
        ret = {
            "total": len(results),
            "validation": statistics,
        }
        res = requests.post(
            self.controller_address + "/calculate_overall",
            json=CalculateOverallRequest(name=self.name, results=results).dict(),
        )
        if res.status_code != 200:
            raise TaskNetworkException(res.text)
        ret["custom"] = res.json()
        return ret
    

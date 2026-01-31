import enum
import json
import requests
import runtime
import time

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
                    headers={"Session_id": str(sid)},
                    json={"session_id": sid},
                    timeout=30,
                )
            except Exception:
                pass

        # ------------------------
        # Transcript aggregation for FC protocol
        # ------------------------
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

        # ------------------------
        # Legacy protocol fallback (some tasks may still return output/history)
        # ------------------------
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
                        headers={"Session_id": str(sid)},
                        json=InteractRequest(session_id=sid, agent_response=response).dict(),
                    )
                except Exception as e:
                    return TaskClientOutput(
                        error=TaskError.NETWORK_ERROR.value,
                        info=str(e),
                        output=latest_result.get("output") if isinstance(latest_result, dict) else None,
                    )

                if interact_resp.status_code != 200:
                    _cancel()
                    return TaskClientOutput(
                        error=TaskError.INTERACT_FAILED.value,
                        info=interact_resp.text,
                        output=latest_result.get("output") if isinstance(latest_result, dict) else None,
                    )

                result = interact_resp.json()
                latest_result = result

            return TaskClientOutput(output=result["output"])

        # ------------------------
        # FC protocol (os-std and other AgentBench FC tasks)
        # start_sample returns {"messages": [...], "tools": [...]}
        # interact expects body {"messages":[{"role":"assistant", ...}]}
        # ------------------------
        max_turns = 200
        for _ in range(max_turns):
            messages = conversation_messages

            # Convert FC messages -> old AgentClient history (user/agent only)
            history = []
            history.append(
                {
                    "role": "user",
                    "content": (
                        "You must respond ONLY with valid JSON.\n"
                        "Schema:\n"
                        "{\"name\":\"bash_action|answer_action|finish_action\","
                        "\"arguments\":{},\"thought\":\"\"}\n"
                        "For bash_action arguments must be: {\"command\":\"...\"}\n"
                        "For answer_action arguments must be: {\"answer\":\"...\"}\n"
                        "For finish_action arguments must be: {}\n"
                        "Do not output any extra text outside JSON."
                    ),
                }
            )

            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if content is None:
                    continue
                if role == "assistant":
                    history.append({"role": "agent", "content": content})
                else:
                    # fold system/user/tool -> "user" since AgentClient only supports user/agent
                    history.append({"role": "user", "content": content})

            try:
                content = agent.inference(history)
            except AgentContextLimitException:
                _cancel()
                return TaskClientOutput(
                    error=TaskError.AGENT_FAILED.value,
                    info="Agent context limit",
                    output=None,
                )
            except Exception as e:
                _cancel()
                return TaskClientOutput(
                    error=TaskError.AGENT_FAILED.value,
                    info=str(e),
                    output=None,
                )

            tool_calls = None
            assistant_content = content

            try:
                obj = json.loads(content)
                if isinstance(obj, dict) and "name" in obj:
                    name = obj["name"]
                    arguments = obj.get("arguments", {})
                    thought = obj.get("thought", "")

                    tool_calls = [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                    assistant_content = thought
            except Exception:
                pass

            msg = {"role": "assistant", "content": assistant_content}
            if tool_calls is not None:
                msg["tool_calls"] = tool_calls

            # record outgoing assistant message in transcript
            _add_messages([msg])

            interact_body = {"messages": [msg]}

            try:
                interact_resp = requests.post(
                    self.controller_address + "/interact",
                    headers={"Session_id": str(sid)},
                    json=interact_body,
                )
            except Exception as e:
                _cancel()
                return TaskClientOutput(
                    error=TaskError.NETWORK_ERROR.value,
                    info=str(e),
                    output=None,
                )

            if interact_resp.status_code != 200:
                _cancel()
                return TaskClientOutput(
                    error=TaskError.INTERACT_FAILED.value,
                    info=interact_resp.text,
                    output=None,
                )

            result = interact_resp.json()
            latest_result = result

            # record incoming messages delta in transcript
            if isinstance(result, dict):
                _add_messages(result.get("messages", []))

            # Completion detection for FC responses
            if isinstance(result, dict):
                status = result.get("status")
                if status and status != "running":
                    final_result = dict(result)
                    final_result["messages"] = conversation_messages
                    try:
                        return TaskClientOutput(
                            output=TaskOutput(index=index, status=SampleStatus(status), result=final_result, history=None)
                        )
                    except Exception:
                        return TaskClientOutput(
                            output=TaskOutput(index=index, status=SampleStatus.UNKNOWN, result=final_result, history=None)
                        )

                if result.get("finish") is True:
                    final_result = dict(result)
                    final_result["messages"] = conversation_messages
                    return TaskClientOutput(
                        output=TaskOutput(index=index, status=SampleStatus.COMPLETED, result=final_result, history=None)
                    )

                # Sometimes controller may return legacy output after FC loop
                if "output" in result:
                    return TaskClientOutput(output=result["output"])

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

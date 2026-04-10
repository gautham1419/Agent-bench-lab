# 框架入口

[🌏English](Entrance_en.md)

框架主要的入口是：

- `src.server.task_controller`: 用于手动启动task_controller。
- `src.start_task`: 用于启动task_worker。
- `src.assigner`: 用于启动评测。
- `src.server.task_worker`: 用于手动启动task_worker。

## src.server.task_controller

task_controller是task server的核心，用于管理所有的task_worker。
task_controller应该是最先启动的，且推荐常开，如无必要也建议全局唯一。
task_controller默认运行在5000端口，也可以通过`--port -p`参数指定。
所有接口有统一的前缀`/api/`。

一个启动task_controller并指定其运行在3000端口的示例：

```bash
python -m src.server.task_controller -p 3000
```

task_controller有以下几个用于监控的接口：

| 接口             | 方法   | 参数 | 说明                                                   |
|----------------|------|----|------------------------------------------------------|
| /list_workers  | GET  | 无  | 返回所有的task_worker                                     |
| /list_sessions | GET  | 无  | 返回所有的session                                         |
| /sync_all      | POST | 无  | 同步所有的task_worker上正在运行的session，如controller意外重启应先调用此接口 |
| /cancel_all    | POST | 无  | 取消所有的task_worker上正在运行的session                        |

## src.start_task

start_task是用于启动task_worker的脚本，其主要功能是读取配置文件并启动task_worker。
start_task的配置文件是`configs/start_task.yaml`，具体详见配置文件介绍。

start_task的参数如下：

- `[--config CONFIG]`: 指定要读取的配置文件，默认为`configs/start_task.yaml`，通常没有必要更改。
- `[--start | -s [TASK_NAME NUM [TASK_NAME NUM ...]]]`: 指定要启动的任务，格式为`TASK_NAME NUM`，其中`TASK_NAME`
  是任务名称，`NUM`是需要启动的worker的个数，如此参数被指定则将覆盖**所有**配置文件中的设置。
- `[--auto-controller | -a]`: 指定是否自动启动task_controller，默认为否。
- `[--base-port | -p PORT]`:
  指定task_worker的基础端口，默认为5001，task_worker将从PORT开始依次启动task_worker。如若共有N个task_worker，那么task_worker的端口将从PORT到PORT+N-1。

## src.assigner

assigner是用于启动评测的脚本，其主要功能是读取配置文件并启动评测，并将结果实时保存在指定的输出文件夹中。

assigner的参数如下：

- `[--config CONFIG]`: 指定要读取的配置文件，默认为`configs/assignments/default.yaml`。
- `[--auto-retry]`: 自动重新测试失败的样例

如配置文件中的`output`字段的值中含有`{TIMESTAMP}`，则此处将会被替换为当前时间并继续后续的操作（即相同的配置文件可能会有不同的输出文件夹）。

如果配置中`output`字段指定的目录已经存在，则assigner将会尝试从此文件夹中读取已有的评测结果，在此基础上继续评测。

assigner**每次**启动都会将读取的配置文件解析并存储到`output`字段指定的目录中，**如目录中已有配置文件，该文件将被覆盖**。

## src.server.task_worker

一个task_worker对应了一个任务进程，同样的任务可以有多个task_worker。
如无必要，**不推荐**手动启动task_worker，而是通过`src.start_task`启动。

task_worker的参数如下：

- `NAME` 任务名称，用于指定要启动的任务。
- `[--config | -c CONFIG]` 指定要读取的配置文件，默认为`configs/tasks/task_assembly.yaml`。
- `[--port | -p PORT]` 指定task_worker的端口，默认为5001。
- `[--controller | -C ADDRESS]` 指定task_controller的地址，默认为http://localhost:5000/api 。
- `[--self ADDRESS]` 指定task_worker的地址，默认为http://localhost:5001/api
  ，此地址将会被task_controller用于与task_worker通信，所以需要确保task_controller能够访问到此地址。

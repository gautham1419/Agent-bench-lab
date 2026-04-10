# 配置系统

[🌏English](Config_cn.md)

## 基本语法

配置系统采用了YAML格式。为了方便配置，我们在基础的YAML语法上做了一些扩展。
`import`, `default`, `overwrite`是我们扩展的关键字。

### import

`import`关键字用于导入其他文件中的配置。例如以下两个写法是等价的：

写法一：

```yaml
# config.yaml
definition:
  def1: something...
  def2: something...
```

写法二：

```yaml
# def1.yaml
def1: something...

# def2.yaml
def2: something...

# config.yaml
definition:
  import:
    - def1.yaml
    - def2.yaml
```

`import`关键字支持字符串或者列表作为值，
分别对应导入单个文件和导入多个文件的情况。

在导入过程中，如果被导入文件中有`import`关键字，
则将先执行被导入文件的`import`。
对于后两个关键字也是如此。

导入过程中如果遇到了键冲突的情况，将尝试递归地合并冲突的键所对应的值。
如果遇到无法合并的情况，则后出现的将覆盖先出现的。

### default

`default`关键字用于指定默认值。例如以下两个写法是等价的：

写法一：

```yaml
definition:
  def1:
    type: int
    value: 1
  def2:
    type: int
    value: 2
  def3:
    type: float
    value: 1.1
```

写法二：

```yaml
definition:
  default:
    type: int
  def1:
    value: 1
  def2:
    value: 2
  def3:
    type: float
    value: 1.1
```

`default`关键字支持字符串、列表或者字典作为值。
config解析器将尝试合并`default`的值和与`default`并列的键所对应的值。
如果遇到无法合并的情况，则`default`关键字下的值具有更低的优先级。

### overwrite

`overwrite`关键字的用法和`default`类似，
只不过在遇到冲突情况时`overwrite`关键字下的值具有更高的优先级。
这个关键字常与`import`联用，用于统一设置这一配置文件下所要求的值。

## 配置文件

配置文件的主要目录结构如下：

```
configs
├── assignments
│   ├── definition.yaml
│   ├── default.yaml
│   └── ...
├── agents
├── tasks
│   ├── task_assembly.yaml
│   └── ...
└── start_task.yaml
```

### assignments

`assignments`目录下存放了所有的任务配置文件。
其中`definition.yaml`集合了所有的任务定义和模型定义。

单个任务配置文件主要需要以下字段：

- `definition`: 通常import自`definition.yaml`，用于定义任务和模型。
- `concurrency`: 用于定义模型的最大并行数。
- `assignments`: 接受多个`assignment`，用于定义任务的具体分配。
- `output`: 用于定义输出文件的路径。

单个`assignment`需要两个字段：

- `agents`: 此任务需要运行的agent的名称。
- `tasks`: 此任务需要运行的task的名称。

### agents

`agents`目录下存放了所有的agent配置文件。
配置中键是agent的名称，值是agent的配置。
单个agent配置需要以下字段：

- `module`: 定义对应的agent client模块。
- `parameters`: 定义需要传入对应模块的参数。

### tasks

`tasks`目录下存放了所有的task配置文件。
其中`task_assembly.yaml`集合了所有的task定义。
如果只是想运行现有的任务，一般不需要修改此目录下的文件。

与agent配置类似，键是task的名称，值是task的配置。
单个task配置需要以下字段：

- `module`: 定义对应的task模块。
- `parameters`: 定义需要传入对应模块的参数。

### start_task.yaml

这个配置文件用于与`src.start_task`配合，自动化批量启动task_worker。
这个文件的字段如下：

- `definition`: 用于定义任务，通常import自`task_assembly.yaml`。
- `start(Optional)`: 用于指定需要启动的任务，键是任务名称，值是需要启动的worker的个数。
- `controller_address(Optional)`: 用于指定controller的地址，默认http://localhost:5000/api/

# CropAbility Java项目开发指南

## 项目概述

CropAbility是一个基于Java和JNI的高性能计算项目，主要用于集成Triton GPU加速算子和PyTorch模型推理。

## 项目结构

```
CropAbility/
├── pom.xml                     # Maven项目配置
├── build.sh                    # 构建脚本
├── README.md                   # 项目说明
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/triton/
│   │   │       ├── TritonJNI.java          # JNI接口类
│   │   │       ├── TritonJNIWrapper.java   # JNI封装类  
│   │   │       └── TritonOperator.java     # PyTorch集成类
│   │   ├── resources/
│   │   │   ├── simplelogger.properties     # 日志配置
│   │   │   └── triton.properties           # 应用配置
│   │   └── python/                         # Python模块
│   └── test/
│       └── java/
│           └── com/example/triton/
│               ├── TritonJNIWrapperTest.java
│               └── TritonOperatorTest.java
└── target/                     # 构建输出目录
```

## 核心组件

### 1. TritonJNI.java
- JNI原生接口类
- 负责加载C++动态库
- 提供模型加载、运算执行、资源释放等原生方法

### 2. TritonJNIWrapper.java  
- JNI的高级封装类
- 提供更友好的Java API
- 包含错误处理和资源管理
- 支持批量处理

### 3. TritonOperator.java
- PyTorch集成类
- 基于PyTorch Java API
- 支持GPU计算和张量操作

## 环境要求

- Java 11+
- Maven 3.6+
- CUDA 11.0+ (GPU加速)
- PyTorch 1.13+ (可选)

## 构建和运行

### 使用构建脚本（推荐）

```bash
# 完整构建
./build.sh

# 单独执行某个步骤
./build.sh clean      # 清理
./build.sh compile    # 编译
./build.sh test       # 测试
./build.sh package    # 打包
./build.sh jni        # 生成JNI头文件
```

### 使用Maven命令

```bash
# 清理和编译
mvn clean compile

# 运行测试
mvn test

# 打包
mvn package

# 生成JNI头文件
mvn native:javah
```

## JNI开发

### 1. 生成头文件

```bash
./build.sh jni
```

生成的头文件位于：`target/native/javah/com_example_triton_TritonJNI.h`

### 2. 实现C++代码

基于生成的头文件实现对应的C++函数：

```cpp
#include "com_example_triton_TritonJNI.h"
#include <jni.h>

JNIEXPORT jlong JNICALL Java_com_example_triton_TritonJNI_loadModel
  (JNIEnv *env, jobject obj, jstring modelPath) {
    // 实现模型加载逻辑
    return (jlong)model_ptr;
}

// 其他JNI函数实现...
```

### 3. 编译动态库

```bash
g++ -shared -fPIC -O3 \
    -I$JAVA_HOME/include \
    -I$JAVA_HOME/include/linux \
    -o libtriton_jni.so \
    native_triton_wrapper.cpp
```

## API使用示例

### 使用JNI版本

```java
try (TritonJNIWrapper triton = new TritonJNIWrapper("model.pt")) {
    float[] input1 = {1.0f, 2.0f, 3.0f};
    float[] input2 = {4.0f, 5.0f, 6.0f};
    
    float[] result = triton.add(input1, input2);
    System.out.println("结果: " + Arrays.toString(result));
}
```

### 使用PyTorch版本

```java
try (TritonOperator triton = new TritonOperator("model.pt")) {
    float[] input1 = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] input2 = {5.0f, 6.0f, 7.0f, 8.0f};
    long[] shape = {2, 2};
    
    float[] result = triton.add(input1, input2, shape);
    System.out.println("GPU计算结果: " + Arrays.toString(result));
}
```

## 性能优化

1. **批量处理**: 使用`batchAdd()`方法处理多个数据
2. **GPU内存管理**: 及时释放GPU张量资源
3. **模型缓存**: 避免重复加载相同模型
4. **异步处理**: 考虑使用CompletableFuture进行异步计算

## 故障排除

### 常见问题

1. **UnsatisfiedLinkError**: 
   - 检查动态库是否正确编译和部署
   - 确认库文件路径在`java.library.path`中

2. **OutOfMemoryError**:
   - 增加JVM堆内存: `-Xmx4g`
   - 优化GPU内存使用

3. **CUDA相关错误**:
   - 检查CUDA版本兼容性
   - 确认GPU驱动程序

### 调试模式

```bash
# 启用详细日志
export JAVA_OPTS="-Dorg.slf4j.simpleLogger.defaultLogLevel=DEBUG"

# 启用JNI检查
export JAVA_OPTS="$JAVA_OPTS -Xcheck:jni"

java $JAVA_OPTS -jar target/CropAbility-1.0.0.jar
```

## 贡献指南

1. Fork项目
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

package com.example.triton;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Logger;

/**
 * CropAbility — TorchScript 模型 Java 集成层
 *
 * <p>此类负责从 Python 端（cropability.models.torchscript）加载 TorchScript 模型，
 * 并提供 Java 调用接口。真正的 GPU 计算在 Python/Triton 端完成，
 * Java 层仅做数据格式转换与模型调用。</p>
 *
 * <p>与 cropability Python 包的集成：
 * <pre>
 *   # Python 端导出
 *   cropability export --model add --output model.pt
 *   cropability export --model embedding --output embed.pt
 * </pre>
 * </p>
 */
public class TritonIntegration implements AutoCloseable {

    private static final Logger logger = Logger.getLogger(TritonIntegration.class.getName());

    /** 默认模型路径（cropability export 的输出位置）。*/
    public static final String DEFAULT_MODEL_PATH = "model.pt";

    private Module model;
    private final String modelPath;

    // ------------------------------------------------------------------
    // 构造与初始化
    // ------------------------------------------------------------------

    /**
     * 从指定路径加载 TorchScript 模型。
     *
     * @param modelPath 由 {@code cropability export} 命令生成的 .pt 文件路径
     * @throws RuntimeException 若模型文件不存在或加载失败
     */
    public TritonIntegration(String modelPath) {
        this.modelPath = modelPath;
        Path p = Paths.get(modelPath);
        if (!Files.exists(p)) {
            throw new RuntimeException(
                "模型文件不存在: " + modelPath +
                "\n请先运行: cropability export --model add --output " + modelPath
            );
        }
        try {
            logger.info("加载 TorchScript 模型: " + modelPath);
            this.model = Module.load(modelPath);
            logger.info("模型加载成功。");
        } catch (Exception e) {
            throw new RuntimeException("模型加载失败: " + e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------------
    // 核心推理接口
    // ------------------------------------------------------------------

    /**
     * 调用 AddModule 执行逐元素加法（向量）。
     *
     * @param input1 第一个浮点数组
     * @param input2 第二个浮点数组（长度需与 input1 相同）
     * @return 逐元素之和
     */
    public float[] add(float[] input1, float[] input2) {
        if (input1.length != input2.length) {
            throw new IllegalArgumentException(
                "输入数组长度不一致: " + input1.length + " vs " + input2.length
            );
        }
        long[] shape = {input1.length};
        Tensor t1 = Tensor.fromBlob(input1, shape);
        Tensor t2 = Tensor.fromBlob(input2, shape);
        IValue result = model.forward(IValue.from(t1), IValue.from(t2));
        return result.toTensor().getDataAsFloatArray();
    }

    /**
     * 批量加法：对多组向量对逐一调用 {@link #add(float[], float[])}。
     *
     * @param batch1 批量输入一（行向量数组）
     * @param batch2 批量输入二（行向量数组）
     * @return 批量结果
     */
    public float[][] batchAdd(float[][] batch1, float[][] batch2) {
        if (batch1.length != batch2.length) {
            throw new IllegalArgumentException("批次大小必须相同");
        }
        float[][] results = new float[batch1.length][];
        for (int i = 0; i < batch1.length; i++) {
            results[i] = add(batch1[i], batch2[i]);
        }
        return results;
    }

    // ------------------------------------------------------------------
    // 验证与基准
    // ------------------------------------------------------------------

    /**
     * 验证加法结果的数值正确性（与 Java 本地计算对比）。
     *
     * @param input1    输入数组 1
     * @param input2    输入数组 2
     * @param tolerance 允许的最大绝对误差
     * @return 所有元素均在容差范围内则返回 true
     */
    public boolean validateResult(float[] input1, float[] input2, float tolerance) {
        float[] result = add(input1, input2);
        for (int i = 0; i < result.length; i++) {
            float expected = input1[i] + input2[i];
            if (Math.abs(result[i] - expected) > tolerance) {
                logger.warning(String.format(
                    "验证失败 @ index %d: got=%.6f expected=%.6f", i, result[i], expected
                ));
                return false;
            }
        }
        return true;
    }

    /**
     * 执行性能基准测试，返回每次推理的平均毫秒数。
     *
     * @param size 向量长度
     * @param runs 测试轮数（前 3 轮用于预热，不计入统计）
     * @return 平均推理时间（毫秒）
     */
    public double benchmarkPerformance(int size, int runs) {
        float[] a = new float[size];
        float[] b = new float[size];
        for (int i = 0; i < size; i++) {
            a[i] = (float) Math.random();
            b[i] = (float) Math.random();
        }
        // 预热
        for (int i = 0; i < 3; i++) {
            add(a, b);
        }
        long total = 0;
        for (int i = 0; i < runs; i++) {
            long t0 = System.nanoTime();
            add(a, b);
            total += System.nanoTime() - t0;
        }
        double avgMs = total / (double) runs / 1_000_000.0;
        logger.info(String.format("Benchmark size=%d runs=%d: avg=%.3f ms", size, runs, avgMs));
        return avgMs;
    }

    // ------------------------------------------------------------------
    // 生命周期
    // ------------------------------------------------------------------

    /** @return 当前加载的模型文件路径 */
    public String getModelPath() {
        return modelPath;
    }

    @Override
    public void close() {
        model = null;
        logger.info("TritonIntegration closed.");
    }

    // ------------------------------------------------------------------
    // 独立入口（集成测试）
    // ------------------------------------------------------------------

    /**
     * 独立集成测试入口。
     *
     * <p>用法：
     * <pre>
     *   # 先导出模型
     *   cropability export --model add --output model.pt
     *   # 再运行 Java 集成测试
     *   java -cp ... com.example.triton.TritonIntegration [model_path]
     * </pre>
     * </p>
     */
    public static void main(String[] args) {
        String modelPath = args.length > 0 ? args[0] : DEFAULT_MODEL_PATH;
        logger.info("=== CropAbility TorchScript 集成测试 ===");
        logger.info("模型路径: " + modelPath);

        try (TritonIntegration integration = new TritonIntegration(modelPath)) {

            // 1. 基本功能
            logger.info("\n--- 1. 基本加法 ---");
            float[] x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] y = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
            float[] out = integration.add(x, y);
            logger.info("x:      " + Arrays.toString(x));
            logger.info("y:      " + Arrays.toString(y));
            logger.info("x + y:  " + Arrays.toString(out));

            // 2. 数值验证
            logger.info("\n--- 2. 数值验证 ---");
            boolean allOk = true;
            for (int sz : new int[]{64, 1024, 65536}) {
                float[] a = new float[sz];
                float[] b = new float[sz];
                for (int i = 0; i < sz; i++) {
                    a[i] = (float) Math.random() * 100;
                    b[i] = (float) Math.random() * 100;
                }
                boolean ok = integration.validateResult(a, b, 1e-5f);
                logger.info(String.format("  size=%-6d  %s", sz, ok ? "PASS" : "FAIL"));
                allOk &= ok;
            }

            // 3. 批处理
            logger.info("\n--- 3. 批处理 ---");
            float[][] bx = {{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}};
            float[][] by = {{1f, 1f, 1f}, {2f, 2f, 2f}, {3f, 3f, 3f}};
            float[][] br = integration.batchAdd(bx, by);
            for (int i = 0; i < br.length; i++) {
                logger.info(String.format("  batch[%d]: %s", i, Arrays.toString(br[i])));
            }

            // 4. 性能基准
            logger.info("\n--- 4. 性能基准 ---");
            for (int sz : new int[]{1_000, 100_000, 1_000_000}) {
                double ms = integration.benchmarkPerformance(sz, 20);
                logger.info(String.format("  size=%-8d  %.3f ms/call", sz, ms));
            }

            logger.info("\n=== 集成测试完成，验证全部" + (allOk ? "通过" : "存在失败项") + " ===");

        } catch (RuntimeException e) {
            logger.severe("集成测试失败: " + e.getMessage());
            logger.info("\n解决方案：");
            logger.info("  1. 安装 Python 依赖: pip install -r requirements.txt");
            logger.info("  2. 导出模型: cropability export --model add --output model.pt");
            logger.info("  3. 确认 libjnitorch.so 已加入 java.library.path");
            System.exit(1);
        }
    }
}

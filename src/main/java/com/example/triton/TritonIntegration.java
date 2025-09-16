package com.example.triton;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.logging.Logger;

/**
 * Triton算子Java集成类 - 统一的TorchScript模型调用接口
 * 
 * 此类专门负责调用Python端导出的TorchScript模型，不重复实现算子逻辑。
 * Python端已实现: Triton GPU算子 + TorchScript导出
 * Java端职责: 模型加载 + 数据转换 + 结果处理
 */
public class TritonIntegration implements AutoCloseable {
    
    private static final Logger logger = Logger.getLogger(TritonIntegration.class.getName());
    private Module model;
    private final String modelPath;
    
    /**
     * 构造函数 - 加载Python端导出的TorchScript模型
     * @param modelPath TorchScript模型文件路径（由Python端生成）
     */
    public TritonIntegration(String modelPath) {
        this.modelPath = modelPath;
        try {
            logger.info("加载TorchScript模型: " + modelPath);
            this.model = Module.load(modelPath);
            logger.info("✓ 模型加载成功 - 包含Python端Triton算子实现");
        } catch (Exception e) {
            throw new RuntimeException("模型加载失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 执行Triton加法运算
     * 注意：这里只做数据转换和模型调用，真正的算子逻辑在Python端的TorchScript模型中
     * 
     * @param input1 第一个输入数组
     * @param input2 第二个输入数组
     * @return 计算结果（由Python端Triton算子计算）
     */
    public float[] add(float[] input1, float[] input2) {
        if (input1.length != input2.length) {
            throw new IllegalArgumentException("输入数组长度必须相同");
        }
        
        try {
            // 1. Java数据 → PyTorch张量
            long[] shape = {input1.length};
            Tensor tensor1 = Tensor.fromBlob(input1, shape);
            Tensor tensor2 = Tensor.fromBlob(input2, shape);
            
            // 2. 调用TorchScript模型（包含Python端的Triton算子）
            IValue result = model.forward(IValue.from(tensor1), IValue.from(tensor2));
            
            // 3. PyTorch张量 → Java数据
            return result.toTensor().getDataAsFloatArray();
            
        } catch (Exception e) {
            throw new RuntimeException("Triton算子执行失败: " + e.getMessage(), e);
        }
    }
    
    /**
     * 批量处理接口
     * @param batch1 批量输入1
     * @param batch2 批量输入2
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
    
    /**
     * 验证计算正确性
     * @param input1 输入数组1  
     * @param input2 输入数组2
     * @param tolerance 容差
     * @return 是否验证通过
     */
    public boolean validateResult(float[] input1, float[] input2, float tolerance) {
        float[] result = add(input1, input2);
        
        for (int i = 0; i < result.length; i++) {
            float expected = input1[i] + input2[i];
            if (Math.abs(result[i] - expected) > tolerance) {
                logger.warning(String.format("验证失败 - 索引 %d: 实际值 %.6f, 期望值 %.6f", 
                    i, result[i], expected));
                return false;
            }
        }
        return true;
    }
    
    /**
     * 性能基准测试
     * @param size 测试数据大小
     * @param runs 运行次数  
     * @return 平均执行时间（毫秒）
     */
    public double benchmarkPerformance(int size, int runs) {
        logger.info(String.format("性能测试 - 大小: %d, 运行次数: %d", size, runs));
        
        // 准备测试数据
        float[] input1 = new float[size];
        float[] input2 = new float[size];
        for (int i = 0; i < size; i++) {
            input1[i] = (float) Math.random();
            input2[i] = (float) Math.random();
        }
        
        // 预热
        for (int i = 0; i < 3; i++) {
            add(input1, input2);
        }
        
        // 计时测试
        long totalTime = 0;
        for (int i = 0; i < runs; i++) {
            long startTime = System.nanoTime();
            add(input1, input2);
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        
        double averageTimeMs = totalTime / (double) runs / 1_000_000.0;
        logger.info(String.format("平均执行时间: %.3f ms", averageTimeMs));
        return averageTimeMs;
    }
    
    /**
     * 获取模型信息
     */
    public String getModelPath() {
        return modelPath;
    }
    
    /**
     * 释放资源
     */
    @Override
    public void close() {
        if (model != null) {
            model = null; // PyTorch Java API会自动回收资源
            logger.info("模型资源已释放");
        }
    }
    
    /**
     * 主测试方法 - 完整的集成测试
     */
    public static void main(String[] args) {
        String modelPath = "src/main/python/triton_add_model.pt";
        logger.info("=== Triton Java集成测试 ===");
        logger.info("模型路径: " + modelPath);
        
        try (TritonIntegration triton = new TritonIntegration(modelPath)) {
            
            // 测试1: 基本功能
            logger.info("\n--- 测试1: 基本功能 ---");
            float[] input1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] input2 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float[] result = triton.add(input1, input2);
            
            logger.info("输入1: " + Arrays.toString(input1));
            logger.info("输入2: " + Arrays.toString(input2)); 
            logger.info("结果:  " + Arrays.toString(result));
            
            // 测试2: 正确性验证
            logger.info("\n--- 测试2: 正确性验证 ---");
            int[] testSizes = {10, 100, 1000};
            boolean allPassed = true;
            
            for (int size : testSizes) {
                float[] test1 = new float[size];
                float[] test2 = new float[size];
                
                for (int i = 0; i < size; i++) {
                    test1[i] = (float) Math.random() * 10;
                    test2[i] = (float) Math.random() * 10;
                }
                
                boolean passed = triton.validateResult(test1, test2, 1e-6f);
                logger.info(String.format("大小 %d 验证: %s", size, passed ? "✓ 通过" : "✗ 失败"));
                allPassed &= passed;
            }
            
            // 测试3: 批量处理
            logger.info("\n--- 测试3: 批量处理 ---");
            float[][] batch1 = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
            float[][] batch2 = {{1.0f, 1.0f, 1.0f}, {2.0f, 2.0f, 2.0f}};
            float[][] batchResults = triton.batchAdd(batch1, batch2);
            
            logger.info("批量处理完成，处理了 " + batchResults.length + " 个批次");
            for (int i = 0; i < batchResults.length; i++) {
                logger.info("批次 " + (i + 1) + " 结果: " + Arrays.toString(batchResults[i]));
            }
            
            // 测试4: 性能基准
            logger.info("\n--- 测试4: 性能基准 ---");
            int[] benchmarkSizes = {1000, 10000, 100000};
            for (int size : benchmarkSizes) {
                double avgTime = triton.benchmarkPerformance(size, 10);
                logger.info(String.format("大小 %d: 平均耗时 %.3f ms", size, avgTime));
            }
            
            // 总结
            logger.info("\n=== 测试总结 ===");
            logger.info("✓ 基本功能测试: 完成");
            logger.info("✓ 正确性验证: " + (allPassed ? "全部通过" : "部分失败"));
            logger.info("✓ 批量处理测试: 完成");  
            logger.info("✓ 性能基准测试: 完成");
            
            if (allPassed) {
                logger.info("🎉 Python→Java集成成功！Triton算子在Java中正常运行");
            } else {
                logger.warning("⚠️ 部分测试失败，请检查Python端模型导出");
            }
            
        } catch (Exception e) {
            logger.severe("集成测试失败: " + e.getMessage());
            e.printStackTrace();
            logger.info("\n💡 常见问题排查:");
            logger.info("1. 确认Python端已导出TorchScript模型: python pgl_main.py export");
            logger.info("2. 确认模型文件路径正确");
            logger.info("3. 确认PyTorch Java库已正确配置");
        }
    }
}

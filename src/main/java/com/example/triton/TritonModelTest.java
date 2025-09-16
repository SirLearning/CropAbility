package com.example.triton;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Triton TorchScript模型测试类
 * 测试从Python导出的TorchScript模型在Java中的运行
 */
public class TritonModelTest {
    
    private static final Logger logger = Logger.getLogger(TritonModelTest.class.getName());
    private Module model;
    private String modelPath;
    
    public TritonModelTest(String modelPath) {
        this.modelPath = modelPath;
        loadModel();
    }
    
    /**
     * 加载TorchScript模型
     */
    private void loadModel() {
        try {
            logger.info("正在加载模型: " + modelPath);
            this.model = Module.load(modelPath);
            logger.info("✓ 模型加载成功");
        } catch (Exception e) {
            logger.severe("✗ 模型加载失败: " + e.getMessage());
            throw new RuntimeException("无法加载模型: " + modelPath, e);
        }
    }
    
    /**
     * 执行张量加法运算
     * 
     * @param input1 第一个输入数组
     * @param input2 第二个输入数组
     * @return 计算结果数组
     */
    public float[] executeAdd(float[] input1, float[] input2) {
        if (input1.length != input2.length) {
            throw new IllegalArgumentException("输入数组长度必须相同");
        }
        
        try {
            // 创建PyTorch张量
            long[] shape = {input1.length};
            Tensor tensor1 = Tensor.fromBlob(input1, shape);
            Tensor tensor2 = Tensor.fromBlob(input2, shape);
            
            logger.fine("输入张量1形状: " + Arrays.toString(tensor1.shape()));
            logger.fine("输入张量2形状: " + Arrays.toString(tensor2.shape()));
            
            // 执行模型推理
            IValue result = model.forward(IValue.from(tensor1), IValue.from(tensor2));
            Tensor outputTensor = result.toTensor();
            
            logger.fine("输出张量形状: " + Arrays.toString(outputTensor.shape()));
            
            // 获取结果数据
            float[] output = outputTensor.getDataAsFloatArray();
            
            // 注意：PyTorch Java API 没有显式的close方法
            // 资源会被GC自动回收
            
            return output;
            
        } catch (Exception e) {
            logger.severe("模型执行失败: " + e.getMessage());
            throw new RuntimeException("模型执行出错", e);
        }
    }
    
    /**
     * 批量处理
     * 
     * @param batch1 第一组输入批次
     * @param batch2 第二组输入批次
     * @return 批次结果
     */
    public float[][] executeBatchAdd(float[][] batch1, float[][] batch2) {
        if (batch1.length != batch2.length) {
            throw new IllegalArgumentException("批次大小必须相同");
        }
        
        float[][] results = new float[batch1.length][];
        for (int i = 0; i < batch1.length; i++) {
            results[i] = executeAdd(batch1[i], batch2[i]);
            logger.fine("批次 " + (i + 1) + "/" + batch1.length + " 处理完成");
        }
        
        return results;
    }
    
    /**
     * 验证模型计算正确性
     * 
     * @param input1 输入数组1
     * @param input2 输入数组2
     * @param tolerance 容差
     * @return 是否验证通过
     */
    public boolean validateResult(float[] input1, float[] input2, float tolerance) {
        float[] result = executeAdd(input1, input2);
        
        // 计算预期结果
        float[] expected = new float[input1.length];
        for (int i = 0; i < input1.length; i++) {
            expected[i] = input1[i] + input2[i];
        }
        
        // 验证结果
        for (int i = 0; i < result.length; i++) {
            if (Math.abs(result[i] - expected[i]) > tolerance) {
                logger.warning(String.format("验证失败 - 索引 %d: 实际值 %.6f, 期望值 %.6f, 差异 %.6f", 
                    i, result[i], expected[i], Math.abs(result[i] - expected[i])));
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 性能基准测试
     * 
     * @param size 测试数据大小
     * @param runs 运行次数
     * @return 平均执行时间（毫秒）
     */
    public double benchmarkPerformance(int size, int runs) {
        logger.info(String.format("开始性能测试 - 大小: %d, 运行次数: %d", size, runs));
        
        // 准备测试数据
        float[] input1 = new float[size];
        float[] input2 = new float[size];
        
        for (int i = 0; i < size; i++) {
            input1[i] = (float) Math.random();
            input2[i] = (float) Math.random();
        }
        
        // 预热
        for (int i = 0; i < 3; i++) {
            executeAdd(input1, input2);
        }
        
        // 计时测试
        long totalTime = 0;
        for (int i = 0; i < runs; i++) {
            long startTime = System.nanoTime();
            executeAdd(input1, input2);
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        
        double averageTimeMs = totalTime / (double) runs / 1_000_000.0;
        logger.info(String.format("平均执行时间: %.3f ms", averageTimeMs));
        
        return averageTimeMs;
    }
    
    /**
     * 释放资源
     */
    public void release() {
        if (model != null) {
            // 注意：PyTorch Java API的Module没有显式close方法
            // 资源会被GC自动回收
            model = null;
            logger.info("模型资源已释放");
        }
    }
    
    /**
     * 主测试方法
     */
    public static void main(String[] args) {
        // 设置日志级别
        Logger rootLogger = Logger.getLogger("");
        rootLogger.setLevel(Level.INFO);
        
        String modelPath = "triton_add_model.pt";
        logger.info("=== Triton TorchScript模型Java测试 ===");
        
        TritonModelTest tester = null;
        try {
            tester = new TritonModelTest(modelPath);
            
            // 测试1: 基本功能测试
            logger.info("\n--- 测试1: 基本功能 ---");
            float[] test1_input1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] test1_input2 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            float[] result1 = tester.executeAdd(test1_input1, test1_input2);
            
            logger.info("输入1: " + Arrays.toString(test1_input1));
            logger.info("输入2: " + Arrays.toString(test1_input2));
            logger.info("结果:  " + Arrays.toString(result1));
            
            // 测试2: 正确性验证
            logger.info("\n--- 测试2: 正确性验证 ---");
            boolean[] validationResults = new boolean[3];
            int[] testSizes = {10, 100, 1000};
            
            for (int i = 0; i < testSizes.length; i++) {
                int size = testSizes[i];
                float[] input1 = new float[size];
                float[] input2 = new float[size];
                
                // 填充测试数据
                for (int j = 0; j < size; j++) {
                    input1[j] = (float) Math.random() * 10;
                    input2[j] = (float) Math.random() * 10;
                }
                
                validationResults[i] = tester.validateResult(input1, input2, 1e-6f);
                logger.info(String.format("大小 %d 验证: %s", size, validationResults[i] ? "✓ 通过" : "✗ 失败"));
            }
            
            // 测试3: 批量处理
            logger.info("\n--- 测试3: 批量处理 ---");
            float[][] batch1 = {
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f},
                {7.0f, 8.0f, 9.0f}
            };
            float[][] batch2 = {
                {1.0f, 1.0f, 1.0f},
                {2.0f, 2.0f, 2.0f},
                {3.0f, 3.0f, 3.0f}
            };
            
            float[][] batchResults = tester.executeBatchAdd(batch1, batch2);
            logger.info("批量处理完成，处理了 " + batchResults.length + " 个批次");
            
            for (int i = 0; i < batchResults.length; i++) {
                logger.info("批次 " + (i + 1) + " 结果: " + Arrays.toString(batchResults[i]));
            }
            
            // 测试4: 性能基准测试
            logger.info("\n--- 测试4: 性能基准测试 ---");
            int[] benchmarkSizes = {1000, 10000, 100000};
            
            for (int size : benchmarkSizes) {
                double avgTime = tester.benchmarkPerformance(size, 10);
                logger.info(String.format("大小 %d: 平均耗时 %.3f ms", size, avgTime));
            }
            
            // 总结
            logger.info("\n=== 测试总结 ===");
            logger.info("✓ 基本功能测试: 完成");
            
            boolean allValidationPassed = true;
            for (boolean result : validationResults) {
                allValidationPassed &= result;
            }
            logger.info("✓ 正确性验证: " + (allValidationPassed ? "全部通过" : "部分失败"));
            logger.info("✓ 批量处理测试: 完成");
            logger.info("✓ 性能基准测试: 完成");
            
            if (allValidationPassed) {
                logger.info("🎉 所有测试通过！TorchScript模型在Java中运行正常");
            } else {
                logger.warning("⚠️  部分测试失败，请检查模型实现");
            }
            
        } catch (Exception e) {
            logger.severe("测试执行失败: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (tester != null) {
                tester.release();
            }
        }
    }
    
    // 移除重复的close方法定义
}

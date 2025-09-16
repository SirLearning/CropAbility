package com.example.triton;

import java.util.Arrays;
import java.util.logging.Logger;

/**
 * Triton算子Java调用封装类 - 轻量级别名，委托给TritonIntegration
 * 
 * @deprecated 使用 TritonIntegration 替代，提供更清晰的职责分离
 */
@Deprecated
public class TritonOperator implements AutoCloseable {
    
    private static final Logger logger = Logger.getLogger(TritonOperator.class.getName());
    private final TritonIntegration integration;
    
    /**
     * 构造函数，委托给TritonIntegration
     * @param modelPath TorchScript模型文件路径
     */
    public TritonOperator(String modelPath) {
        logger.warning("TritonOperator已废弃，建议使用TritonIntegration");
        this.integration = new TritonIntegration(modelPath);
    }
    
    /**
     * 执行Triton加法运算 - 委托给TritonIntegration
     */
    public float[] add(float[] input1, float[] input2) {
        return integration.add(input1, input2);
    }
    
    /**
     * 批量处理接口 - 委托给TritonIntegration
     */
    public float[][] batchAdd(float[][] batch1, float[][] batch2) {
        return integration.batchAdd(batch1, batch2);
    }
    
    /**
     * 获取模型信息 - 委托给TritonIntegration
     */
    public String getModelPath() {
        return integration.getModelPath();
    }
    
    /**
     * 释放模型资源 - 委托给TritonIntegration
     */
    @Override
    public void close() {
        if (integration != null) {
            integration.close();
        }
    }
    
    // 简化的测试方法 - 建议使用TritonIntegration.main()替代
    public static void main(String[] args) {
        logger.warning("TritonOperator.main()已废弃，请使用: java TritonIntegration");
        logger.info("运行新的集成测试...");
        
        // 委托给新的集成类
        TritonIntegration.main(args);
    }
}

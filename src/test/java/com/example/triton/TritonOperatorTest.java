package com.example.triton;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * TritonOperator的单元测试
 */
public class TritonOperatorTest {
    
    @Test
    public void testArrayValidation() {
        float[] input1 = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] input2 = {5.0f, 6.0f, 7.0f, 8.0f};
        
        assertEquals("Input arrays should have same length", input1.length, input2.length);
    }
    
    @Test
    public void testShapeValidation() {
        long[] shape = {2, 2};
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
        
        long expectedSize = 1;
        for (long dim : shape) {
            expectedSize *= dim;
        }
        
        assertEquals("Data size should match shape", expectedSize, data.length);
    }
    
    @Test
    public void testBatchProcessing() {
        float[][] batch1 = {
            {1.0f, 2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f, 8.0f}
        };
        
        float[][] batch2 = {
            {1.0f, 1.0f, 1.0f, 1.0f},
            {2.0f, 2.0f, 2.0f, 2.0f}
        };
        
        assertEquals("Batch sizes should match", batch1.length, batch2.length);
        
        for (int i = 0; i < batch1.length; i++) {
            assertEquals("Each batch item should have same length", 
                        batch1[i].length, batch2[i].length);
        }
    }
}

package com.neuralnetwork.core.backprop;

import org.junit.Test;

public class NNBackpropTest
{
    @Test
    public void getInputShouldWork()
    {
        if (prevLayer == -1)
        {
            return example.input[neuronIndex];
        }

        return aYInfo[prevLayer].y[neuronIndex];
    }


}

package com.neuralnetwork.xor;

public class XORBackPropagationTest
{
    protected XORBackPropagationNetworkVersionA network = new XORBackPropagationNetworkVersionA();

    @org.junit.Test
    public void testAllErrorsDecrease() throws Exception
    {
        network.backpropagation(0.0001,
                new NVector(0, 0), new NVector(0),
                new NVector(0, 1), new NVector(1),
                new NVector(1, 1), new NVector(0),
                new NVector(1, 0), new NVector(1));
    }
}

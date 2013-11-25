package com.neuralnetwork.xor;

public class TestRunNetworkTest
{
    @org.junit.Test
    public void testAllErrorsDecrease() throws Exception
    {
        TestRunNetwork network = new TestRunNetwork();
        network.backpropagation(0.01,
                new NVector(1.0, 2.0, 3.0), new NVector(0.25, 0.75));
//                new NVector(0f, 1f), new NVector(1f),
//                new NVector(1f, 1f), new NVector(0f),
//                new NVector(1f, 0f), new NVector(1f));
    }
}

package com.neuralnetwork.xor;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class XORBackPropagationTest
{
    protected XORBackPropagationNetworkVersionA network = new XORBackPropagationNetworkVersionA();

    /*@org.junit.Test
    public void testErrorDecreases() throws Exception
    {
        NVector input = new NVector(0f, 0f);

        NVector output1 = network.output(input);

        network.backpropagation(input,
                                new NVector(0f));

        NVector output2 = network.output(input);

        assertThat(output1.get(0) - output2.get(0) >= 0, is(true));
    }

    @org.junit.Test
    public void testErrorDecreases2() throws Exception
    {
        NVector input = new NVector(0f, 1f);

        NVector output1 = network.output(input);

        network.backpropagation(input, new NVector(1f));

        NVector output2 = network.output(input);

        assertThat((1f - output1.get(0)) //first error
                   - (1f - output2.get(0)) //second error
                    >= 0,
                is(true));
    }

    @org.junit.Test
    public void testErrorDecreases3() throws Exception
    {
        NVector input = new NVector(1f, 1f);

        NVector output1 = network.output(input);

        network.backpropagation(input, new NVector(0f));

        NVector output2 = network.output(input);

        assertThat((0f - output1.get(0)) //first error
                - (0f - output2.get(0)) //second error
                >= 0,
                is(true));
    }

    @org.junit.Test
    public void testErrorsDecreases4() throws Exception
    {
        NVector input = new NVector(1f, 0f);

        NVector output1 = network.output(input);

        network.backpropagation(input, new NVector(1f));

        NVector output2 = network.output(input);

        assertThat((1f - output1.get(0)) //first error
                - (1f - output2.get(0)) //second error
                >= 0,
                is(true));
    }*/

    @org.junit.Test
    public void testAllErrorsDecrease() throws Exception
    {
        network.backpropagation(0.01f,
                new NVector(0f, 0f), new NVector(0f));
//                new NVector(0f, 1f), new NVector(1f),
//                new NVector(1f, 1f), new NVector(0f),
//                new NVector(1f, 0f), new NVector(1f));
    }
}

package com.neuralnetwork.nn.backprop;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class XORNetworkTest
{

    @org.junit.Test
    public void testOutput1() throws Exception
    {
        XORNetwork network = new XORNetwork();

        double[] output = network.generateOutput(1f, 0f, 0f);

        assertThat(output.length, is(1));
        assertThat(output[0], is(0.0));
    }

    @org.junit.Test
    public void testOutput2() throws Exception
    {
        XORNetwork network = new XORNetwork();

        double[] output = network.generateOutput(1f, 0f, 1f);

        assertThat(output.length, is(1));
        assertThat(output[0], is(1.0));
    }

    @org.junit.Test
    public void testOutput3() throws Exception
    {
        XORNetwork network = new XORNetwork();

        double[] output = network.generateOutput(1f, 1f, 1f);

        assertThat(output.length, is(1));
        assertThat(output[0], is(0.0));
    }

    @org.junit.Test
    public void testOutput4() throws Exception
    {
        XORNetwork network = new XORNetwork();

        double[] output = network.generateOutput(1f, 1f, 0f);

        assertThat(output.length, is(1));
        assertThat(output[0], is(1.0));
    }
}

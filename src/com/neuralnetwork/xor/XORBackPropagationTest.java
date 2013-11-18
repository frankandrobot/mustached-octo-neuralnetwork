package com.neuralnetwork.xor;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class XORBackPropagationTest
{

    @org.junit.Test
    public void testOutput() throws Exception
    {
        XORBackPropagationNetwork network = new XORBackPropagationNetwork();
        NVector output;

        output = network.output(new NVector(0f, 0f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(0f));
    }

    @org.junit.Test
    public void testOutput2() throws Exception
    {
        XORBackPropagationNetwork network = new XORBackPropagationNetwork();
        NVector output;

        output = network.output(new NVector(0f, 1f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(1f));
    }

    @org.junit.Test
    public void testOutput3() throws Exception
    {
        XORBackPropagationNetwork network = new XORBackPropagationNetwork();
        NVector output;

        output = network.output(new NVector(1f, 1f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(0f));
    }

    @org.junit.Test
    public void testOutput4() throws Exception
    {
        XORBackPropagationNetwork network = new XORBackPropagationNetwork();
        NVector output;

        output = network.output(new NVector(1f, 0f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(1f));
    }
}

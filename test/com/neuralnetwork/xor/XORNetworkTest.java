package com.neuralnetwork.xor;

import com.neuralnetwork.core.NVector;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class XORNetworkTest
{

    @org.junit.Test
    public void testOutput() throws Exception
    {
        XORNetwork network = new XORNetwork();
        NVector output;

        output = network.output(new NVector(0f, 0f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(0.0));
    }

    @org.junit.Test
    public void testOutput2() throws Exception
    {
        XORNetwork network = new XORNetwork();
        NVector output;

        output = network.output(new NVector(0f, 1f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(1.0));
    }

    @org.junit.Test
    public void testOutput3() throws Exception
    {
        XORNetwork network = new XORNetwork();
        NVector output;

        output = network.output(new NVector(1f, 1f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(0.0));
    }

    @org.junit.Test
    public void testOutput4() throws Exception
    {
        XORNetwork network = new XORNetwork();
        NVector output;

        output = network.output(new NVector(1f, 0f));

        assertThat(output.size(), is(1));
        assertThat(output.first(), is(1.0));
    }
}

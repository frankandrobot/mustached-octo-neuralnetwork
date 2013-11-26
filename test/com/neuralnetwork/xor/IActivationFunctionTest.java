package com.neuralnetwork.xor;

import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class IActivationFunctionTest
{
    @Test
    public void testSigmoidFunction()
    {
        IActivationFunction sig = new IActivationFunction.SigmoidFunction(2.0);


        assertThat(output(sig, 1.0), is("0.880797"));
        assertThat(output(sig, 3.0), is("0.997527"));
        assertThat(output(sig, 10.0), is("1.000000"));

        assertThat(output(sig, 0.0), is("0.500000"));
        assertThat(output(sig, -1.0), is("0.119203"));
        assertThat(output(sig, -15.0), is("0.000000"));

    }

    @Test
    public void testUnitySigmoidFunction()
    {
        IActivationFunction sig = new IActivationFunction.SigmoidUnityFunction();


        assertThat(output(sig, 0.0), is("0.500000"));
        assertThat(output(sig, 5.0), is("0.993307"));
        assertThat(output(sig, 10.0), is("0.999955"));
        assertThat(output(sig, 50.0), is("1.000000"));

        assertThat(output(sig, -5.0), is("0.006693"));
        assertThat(output(sig, -10.0), is("0.000045"));
        assertThat(output(sig, -50.0), is("0.000000"));

    }

    private String output(IActivationFunction f, double x)
    {
        return String.format("%.6f", f.apply(x));
    }

    @Test
    public void testSigmoidFunctionDeriv()
    {
        IActivationFunction.IDifferentiableFunction sig = new IActivationFunction.SigmoidFunction(2.0);


        assertThat(output(sig, 0.0), is("0.500000"));
        assertThat(output(sig, 3.0), is("0.004933"));
        assertThat(output(sig, 5.0), is("0.000091"));
        assertThat(output(sig, 10.0),is("0.000000"));

        assertThat(output(sig, -3.0), is("0.004933"));
        assertThat(output(sig, -6.0), is("0.000012"));
        assertThat(output(sig, -12.0),is("0.000000"));

    }

    @Test
    public void testUnitySigmoidFunctionDeriv()
    {
        IActivationFunction.IDifferentiableFunction sig = new IActivationFunction.SigmoidUnityFunction();

        assertThat(output(sig, 0.0), is("0.250000"));
        assertThat(output(sig, 3.0), is("0.045177"));
        assertThat(output(sig, 5.0), is("0.006648"));
        assertThat(output(sig, 10.0),is("0.000045"));
        assertThat(output(sig, 20.0),is("0.000000"));

        assertThat(output(sig, -3.0), is("0.045177"));
        assertThat(output(sig, -6.0), is("0.002467"));
        assertThat(output(sig, -20.0), is("0.000000"));

    }

    private String output(IActivationFunction.IDifferentiableFunction f, double x)
    {
        return String.format("%.6f", f.derivative(x));
    }
}

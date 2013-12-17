package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class MNeuronTest
{
    @Test
    public void testConstructor()
    {
        double[] w = {0.1,0.2,0.3};
        MNeuron neuron = new MNeuron(new ActivationFunctions.SigmoidUnityFunction(), w);
        neuron.mWeights.print();

        assertThat(neuron.mWeights.unsafe_get(0,0), is(0.1));
        assertThat(neuron.mWeights.unsafe_get(1,0), is(0.2));
    }

    @Test
    public void testRawoutput() throws Exception
    {
        double[] w = {0.1,0.2,0.3};
        MNeuron neuron = new MNeuron(new ActivationFunctions.SigmoidUnityFunction(), w);

        DenseMatrix64F input = new DenseMatrix64F(1, 2, true, new double[]{0.2, 0.3});
        double rslt = neuron.rawoutput(input);
        assertThat(rslt, is(0.2*0.1 + 0.3*0.2 + 0.3));
    }

    @Test
    public void testOutput() throws Exception
    {
        double[] w = {0.1,0.2,0.3};
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();
        MNeuron neuron = new MNeuron(phi, w);

        DenseMatrix64F input = new DenseMatrix64F(1, 2, true, new double[]{0.2, 0.3});
        double rslt = neuron.output(input);
        assertThat(rslt, is(phi.apply(0.2*0.1 + 0.3*0.2 + 0.3)));
    }

    @Test
    public void testGetWeight() throws Exception
    {
        double[] w = {0.1,0.2,0.3};
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();
        MNeuron neuron = new MNeuron(phi, w);

        assertThat(neuron.getWeight(0), is(0.1));
        assertThat(neuron.getWeight(1), is(0.2));
        assertThat(neuron.getWeight(2), is(0.3));
    }

    @Test
    public void testSetWeight() throws Exception
    {
        double[] w = {0.1,0.2,0.3};
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();
        MNeuron neuron = new MNeuron(phi, w);

        neuron.setWeight(0, -0.1);
        neuron.setWeight(1, -0.2);
        neuron.setWeight(2, -0.3);

        assertThat(neuron.getWeight(0), is(-0.1));
        assertThat(neuron.getWeight(1), is(-0.2));
        assertThat(neuron.getWeight(2), is(-0.3));
    }
}

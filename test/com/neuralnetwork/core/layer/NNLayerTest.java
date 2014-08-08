package com.neuralnetwork.core.layer;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.neuron.Neuron;
import com.neuralnetwork.helpers.NumberFormatter;
import org.ejml.data.DenseMatrix64F;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class NNLayerTest {

    IActivationFunction.IDifferentiableFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    double[] weights1 = new double[]{0.1f, 0.2f, 0.3f};
    double[] weights2 = new double[]{0.4f, 0.5f, 0.6f};

    NNLayer layer;

    @Before
    public void setup() throws Exception
    {
        Neuron n1 = new Neuron(phi, weights1);
        Neuron n2 = new Neuron(phi, weights2);

        layer = new NNLayerBuilder()
                .setNeurons(n1, n2)
                .build();
    }

    @Test
    public void testGenerateY() throws Exception
    {
        double[] input = new double[] { 1f, 0.7f, 0.8f };

        double[] result = layer.generateY(input);

        double o1 = phi.apply(0.1f*1f + 0.2f*0.7f + 0.3f*0.8f);
        double o2 = phi.apply(0.4f*1f + 0.5f*0.7f + 0.6f*0.8f);

        assertThat(result[0], is(1.0));
        assertThat(NumberFormatter.toStr(result[1]),
                is(NumberFormatter.toStr(o1)));
        assertThat(NumberFormatter.toStr(result[2]),
                is(NumberFormatter.toStr(o2)));
    }

    @Test
    public void testGenerateOutput() throws Exception
    {
        double[] input = new double[] { 1f, 0.7f, 0.8f };

        double[] result = layer.generateOutput(input);

        double o1 = phi.apply(0.1f*1f + 0.2f*0.7f + 0.3f*0.8f);
        double o2 = phi.apply(0.4f*1f + 0.5f*0.7f + 0.6f*0.8f);

        assertThat(NumberFormatter.toStr(result[0]),
                is(NumberFormatter.toStr(o1)));
        assertThat(NumberFormatter.toStr(result[1]),
                is(NumberFormatter.toStr(o2)));

    }

    @Test
    public void testGenerateInducedLocalField() throws Exception
    {
        double[] input = new double[] { 1f, 0.7f, 0.8f };

        double[] result = layer.generateInducedLocalField(input);

        double o1 = 0.1f*1f + 0.2f*0.7f + 0.3f*0.8f;
        double o2 = 0.4f*1f + 0.5f*0.7f + 0.6f*0.8f;

        assertThat(NumberFormatter.toStr(result[0]),
                is(NumberFormatter.toStr(o1)));
        assertThat(NumberFormatter.toStr(result[1]),
                is(NumberFormatter.toStr(o2)));
    }

    @Test
    public void testGetInputDim() throws Exception
    {
        assertThat(layer.getInputDim(),
                is(3));
    }

    @Test
    public void testGetOutputDim() throws Exception {

        assertThat(layer.getOutputDim(),
                is(2));
    }

    @Test
    public void testGetNumberOfNeurons() throws Exception {

        assertThat(layer.getNumberOfNeurons(),
                is(2));
    }

    @Test
    public void testGetWeightMatrix() throws Exception {

        DenseMatrix64F weights = layer.getWeightMatrix();

        for(int col=0; col<weights.numCols; ++col)
            assertThat(weights.get(0,col), is(weights1[0]));

        for(int col=0; col<weights.numCols; ++col)
            assertThat(weights.get(1,col), is(weights2[0]));

    }
}

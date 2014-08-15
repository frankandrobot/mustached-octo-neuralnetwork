package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.neuron.Neuron;
import com.neuralnetwork.helpers.NumberAssert;
import com.neuralnetwork.nn.MultiLayerNN;
import com.neuralnetwork.nn.MultiLayerNNBuilder;
import com.neuralnetwork.nn.layer.NNLayer;
import com.neuralnetwork.nn.layer.NNLayerBuilder;
import org.junit.Before;
import org.junit.Test;

import static com.neuralnetwork.helpers.NumberAssert.*;
import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class NNBackpropTest {

    IActivationFunction.IDifferentiableFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    double[] weights1 = new double[]{0.1f, 0.2f, 0.3f};
    double[] weights2 = new double[]{0.4f, 0.5f};

    Example example = new Example();

    NNBackprop backprop;

    NNLayer layer1;
    NNLayer layer2;

    @Before
    public void setup() throws Exception
    {
        Neuron n1 = new Neuron(phi, weights1);
        Neuron n2 = new Neuron(phi, weights2);

        layer1 = new NNLayerBuilder()
                .setNeurons(n1)
                .build();

        layer2 = new NNLayerBuilder()
                .setNeurons(n2)
                .build();

        MultiLayerNN nn = new MultiLayerNNBuilder()
                .setLayers(layer1, layer2)
                .build();

        backprop = new NNBackprop(layer1, layer2);

        example.input = new double[]{1f, 0.6f, 0.7f};
        example.expected = new double[]{0.8f};

        backprop.example = example;

        backprop.forwardProp();
    }

    @Test
    public void testGo() throws Exception {

    }

    @Test
    public void testForwardProp() throws Exception {

        double[] inducedLocalField0 = layer1.generateInducedLocalField(example.input);

        double[] yInducedLocalField0 = new double[inducedLocalField0.length+1];
        yInducedLocalField0[0] = 1;
        System.arraycopy(inducedLocalField0,0,
                yInducedLocalField0,1,
                inducedLocalField0.length);

        _assert(yInducedLocalField0, backprop.aYInfo[0].yInducedLocalField);


        double[] y0 = layer1.generateY(example.input);
        _assert(y0, backprop.aYInfo[0].y);


        double[] inducedLocalField1 = layer2.generateInducedLocalField(y0);

        double[] yInducedLocalField1 = new double[inducedLocalField1.length+1];
        yInducedLocalField1[0] = 1;
        System.arraycopy(inducedLocalField1,0,
                yInducedLocalField1,1,
                inducedLocalField1.length);

        _assert(yInducedLocalField1, backprop.aYInfo[1].yInducedLocalField);


        double[] y1 = layer2.generateY(y0);
        _assert(y1, backprop.aYInfo[1].y);

    }

    @Test
    public void testBackprop() throws Exception {

    }

    @Test
    public void testConstructGradients() throws Exception {

    }

    @Test
    public void testGradient() throws Exception {

    }

    @Test
    public void testSumGradients() throws Exception {

        double sum1 = backprop.sumGradients(1,1);

        //neuron 1^0 is connected to .... neuron 1^1
    }

    @Test
    public void testUpdateCumulativeLearningTerms() throws Exception
    {

    }

    @Test
    public void testGetY() throws Exception
    {
        backprop.forwardProp();

        double e1 = backprop.getY(-1)[0];
        assertThat(e1, is(1.0));

        double e2 = backprop.getY(-1)[1];
        assertThat(e2, is(example.input[1]));

        double[] y = layer1.generateY(example.input);

        assertThat(toStr(backprop.getY(0)[0]),
                is(toStr(y[0])));

        assertThat(toStr(backprop.getY(0)[1]),
                is(toStr(y[1])));
    }
}

package com.neuralnetwork.nn;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.neuron.Neuron;
import com.neuralnetwork.nn.layer.NNLayer;
import com.neuralnetwork.nn.layer.NNLayerBuilder;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class MultiLayerNNTest {

    IActivationFunction.IDifferentiableFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    double[] weights1 = new double[]{0.1f, 0.2f};
    double[] weights2 = new double[]{0.3f, 0.4f};

    MultiLayerNN nn;

    @Before
    public void setup() throws Exception
    {
        Neuron n1 = new Neuron(phi, weights1);
        Neuron n2 = new Neuron(phi, weights2);

        NNLayer layer1 = new NNLayerBuilder()
                .setNeurons(n1)
                .build();

        NNLayer layer2 = new NNLayerBuilder()
                .setNeurons(n2)
                .build();

        nn = new MultiLayerNNBuilder()
                .setLayers(layer1, layer2)
                .build();
    }

    @Test
    public void testGenerateOutput() throws Exception {

        double[] input = new double[]{ 1f, 0.5f };

        double o1 = phi.apply(0.1f*1f + 0.2f*0.5f);
        double[] y1 = new double[]{ 1f, o1 };

        double o2 = phi.apply(0.3f*1f + 0.4f*o1);

        double[] results = nn.generateOutput(input);

        assertThat(results.length, is(1));
        assertThat(results[0], is(o2));
    }
}

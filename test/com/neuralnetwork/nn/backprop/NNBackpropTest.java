package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.Example;
import com.neuralnetwork.helpers.NVector;
import com.neuralnetwork.nn.MultiLayerNN;
import com.neuralnetwork.nn.MultiLayerNNBuilder;
import com.neuralnetwork.nn.layer.NNLayer;
import com.neuralnetwork.nn.layer.NNLayerBuilder;
import org.junit.Test;

import java.util.Random;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class NNBackpropTest
{

    private String round(double num, int precision)
    {
        return String.format("%"+precision+"g", num);
    }

    @Test
    public void testSingleExample() throws Exception
    {
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

        final long stableSeed = 100012;
        Random r = new Random(stableSeed);

        NNLayer firstLayer = new NNLayerBuilder()
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .build();

        NNLayer secondLayer = new NNLayerBuilder()
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .build();

        MultiLayerNN network = new MultiLayerNNBuilder()
                .setLayers(firstLayer, secondLayer)
                .build();

        Example example = new Example()
                .setInput(1, 0.5, 0.2)
                .setExpected(1, 0.8);

        NNBackprop backprop = new NNBackpropBuilder()
                .setNet(network)
                .setExamples(example)
                .setGamma(0.005)
                .setEta(0.009)
                .build();

        final double errorTol = 0.00001;

        backprop.go(errorTol);

        validateOutput(network, example, errorTol);
    }

    @Test
    public void testTwoExamples() throws Exception
    {
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

        final long stableSeed = 100012;
        Random r = new Random(stableSeed);

        NNLayer firstLayer = new NNLayerBuilder()
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .build();

        NNLayer secondLayer = new NNLayerBuilder()
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .build();

        MultiLayerNN network = new MultiLayerNNBuilder()
                .setLayers(firstLayer, secondLayer)
                .build();

        Example example1 = new Example()
                .setInput(1, 0.5, 0.2)
                .setExpected(1, 0.2);

        Example example2 = new Example()
                .setInput(1, 10, 20)
                .setExpected(1, 0.8);

        NNBackprop backprop = new NNBackpropBuilder()
                .setExamples(example1, example2)
                .setNet(network)
                .setGamma(0.00002)
                .setEta(0.1)
                .setIterations(35000)
                .build();


        final double errorTol = 0.0001;

        backprop.go(errorTol);

        validateOutput(network, example1, errorTol);
        validateOutput(network, example2, errorTol);
    }

    private void validateOutput(MultiLayerNN network, Example example, double errorTol)
    {
        NVector rslt = new NVector(network.generateYoutput(example.input));
        NVector exp = new NVector(example.expected);

        //System.out.format("%nrslt - expected = %s - %s = %s%n", rslt, expected,
        //    rslt.subtract(expected));
        //System.out.println("error: "+rslt.subtract(expected).dotProduct());
        assertThat(rslt.subtract(exp).dotProduct() < errorTol, is(true));
    }
}

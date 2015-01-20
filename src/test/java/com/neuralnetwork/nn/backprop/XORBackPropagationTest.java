package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.INnLayer;
import com.neuralnetwork.helpers.NVector;
import com.neuralnetwork.nn.NN;
import com.neuralnetwork.nn.NNBuilder;
import com.neuralnetwork.nn.layer.NNLayerBuilder;

import java.util.Random;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class XORBackPropagationTest
{
    @org.junit.Test
    public void testLearning() throws Exception
    {
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

        final long stableSeed = 99991000;
        Random r = new Random(stableSeed);

        INnLayer firstLayer = new NNLayerBuilder()
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .setNeuron(phi, r.nextGaussian(), r.nextGaussian(), r.nextGaussian())
                .build();

        INnLayer secondLayer = new NNLayerBuilder()
                .setNeuron(phi,
                        r.nextGaussian(),
                        r.nextGaussian(),
                        r.nextGaussian(),
                        r.nextGaussian())
                .build();

        NN network = new NNBuilder()
                .setLayers(firstLayer, secondLayer)
                .build();

        final double tolerance = 0.0001;

        Example example1 = new Example()
                .setInput(1,0,0)
                .setExpected(1,0.8);

        Example example2 = new Example()
                .setInput(1,1,1)
                .setExpected(1,0.8);

        Example example3 = new Example()
                .setInput(1,1,0)
                .setExpected(1,0.2);
        Example example4 = new Example()
                .setInput(1,0,1)
                .setExpected(1,0.2);

        NNBackprop backprop = new NNBackpropBuilder()
                .setNet(network)
                .setExamples(example1, example2, example3, example4)
                .setGamma(0.00001)
                .setEta(0.3)
                .setIterations(6000)
                .build();

        backprop.go(tolerance);

        output(network.generateOutput(example1.input));
        output(new NVector(network.generateYoutput(example1.input))
                        .subtract(new NVector(example1.expected)).data());

        assertThat(new NVector(network.generateYoutput(example1.input))
                .subtract(new NVector(example1.expected))
                .dotProduct() < tolerance,
                is(true));

        assertThat(new NVector(network.generateYoutput(example2.input))
                .subtract(new NVector(example2.expected))
                .dotProduct() < tolerance,
                is(true));

        assertThat(new NVector(network.generateYoutput(example3.input))
                .subtract(new NVector(example3.expected))
                .dotProduct() < tolerance,
                is(true));

        assertThat(new NVector(network.generateYoutput(example4.input))
                .subtract(new NVector(example4.expected))
                .dotProduct() < tolerance,
                is(true));
    }

    private void output(double val)
    {
        String num = String.format("%.10f", val);
        System.out.println(num);
    }

    private void output(double[] val)
    {
        String tmp = "[";

        for(double v:val)
            tmp += v+",";
        tmp += "]";

        System.out.println(tmp);
    }
}

package com.neuralnetwork.cnn.old;

import com.neuralnetwork.cnn.MNeuron;
import com.neuralnetwork.cnn.old.OldConvolutionMapLayer;
import com.neuralnetwork.cnn.old.OldConvolutionalNetwork;
import com.neuralnetwork.cnn.old.OldFeatureMap;
import com.neuralnetwork.cnn.old.OldSubSamplingMap;
import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import java.util.Random;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class ConvolutionalNetworkBackPropTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    @Test
    public void testGradientsSingleConvolutionMapLayer()
    {
        final DenseMatrix64F trainingInput = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final DenseMatrix64F expected = new DenseMatrix64F(2,2,true, new double[] {
                0.4, 0.6
               ,0.1, 0.8
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);

        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(featureMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        OldConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{trainingInput, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);

        final DenseMatrix64F mImpulseFunction = network.getLayer(0).mImpulseFunction;
        final DenseMatrix64F mInducedField = network.getLayer(0).mInducedLocalField;
        final DenseMatrix64F mGradient = network.getLayer(0).mGradients;

        //calculate gradient at (0,0)
        final double d00 = (expected.get(0,0) - mImpulseFunction.get(0,0)) * phi.derivative( mInducedField.get(0,0) );
        assertThat(d00, is(mGradient.get(0,0)));

        final double d01 = (expected.get(0,1) - mImpulseFunction.get(0,1)) * phi.derivative( mInducedField.get(0,1) );
        assertThat(d01, is(mGradient.get(0,1)));

        final double d11 = (expected.get(1,1) - mImpulseFunction.get(1,1)) * phi.derivative( mInducedField.get(1,1) );
        assertThat(d11, is(mGradient.get(1,1)));

        final double d10 = (expected.get(1,0) - mImpulseFunction.get(1,0)) * phi.derivative( mInducedField.get(1,0) );
        assertThat(d10, is(mGradient.get(1,0)));
    }

    @Test
    public void testGradientsSingleSubsamplingMapLayer()
    {
        final DenseMatrix64F trainingInput = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final DenseMatrix64F expected = new DenseMatrix64F(2,2,true, new double[] {
                0.4, 0.6
               ,0.1, 0.8
        });

        final double[] weights = {0.3, 0.4};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2*2);
        builder.set1DInputSize(4);

        OldFeatureMap featureMap = new OldSubSamplingMap(builder);

        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(featureMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        OldConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{trainingInput, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);

        final DenseMatrix64F mImpulseFunction = network.getLayer(0).mImpulseFunction;
        final DenseMatrix64F mInducedField = network.getLayer(0).mInducedLocalField;
        final DenseMatrix64F mGradient = network.getLayer(0).mGradients;

        //calculate gradient at (0,0)
        final double d00 = (expected.get(0,0) - mImpulseFunction.get(0,0)) * phi.derivative( mInducedField.get(0,0) );
        assertThat(d00, is(mGradient.get(0,0)));

        final double d01 = (expected.get(0,1) - mImpulseFunction.get(0,1)) * phi.derivative( mInducedField.get(0,1) );
        assertThat(d01, is(mGradient.get(0,1)));

        final double d11 = (expected.get(1,1) - mImpulseFunction.get(1,1)) * phi.derivative( mInducedField.get(1,1) );
        assertThat(d11, is(mGradient.get(1,1)));

        final double d10 = (expected.get(1,0) - mImpulseFunction.get(1,0)) * phi.derivative( mInducedField.get(1,0) );
        assertThat(d10, is(mGradient.get(1,0)));
    }

    @Test
    public void testTwoLayerA()
    {
        //construct convolution layer
        final DenseMatrix64F trainingInput = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final DenseMatrix64F expected = new DenseMatrix64F(1,1,true, new double[] {0.4});

        final double[] weights = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};

        //build first layer
        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(3 * 3);
        builder.set1DInputSize(4);

        OldFeatureMap convolutionMap = new OldConvolutionMapLayer(builder);

        assertThat(convolutionMap.getFeatureMap().numCols, is(2));
        assertThat(convolutionMap.getFeatureMap().numRows, is(2));

        //build second layer
        final double[] weights2 = {0.3, 0.4};
        builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights2));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(2);

        OldFeatureMap subsamplingMap = new OldSubSamplingMap(builder);

        assertThat(subsamplingMap.getFeatureMap().numCols, is(1));
        assertThat(subsamplingMap.getFeatureMap().numRows, is(1));

        //build network
        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(convolutionMap, subsamplingMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        OldConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{trainingInput, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);

        //calculate gradient at (0,0) in conv layer (0)
        final DenseMatrix64F mInducedField0 = network.getLayer(0).mInducedLocalField;
        final DenseMatrix64F mGradients0 = network.getLayer(0).mGradients;
        final DenseMatrix64F mGradients1 = network.getLayer(1).mGradients;

        final double d00 = phi.derivative(mInducedField0.get(0,0)) * mGradients1.get(0,0) * weights2[0];

        assertThat(d00, is(mGradients0.get(0, 0)));
    }

    /**
     * first layer is subsampling, second layer is convolution
     */
    @Test
    public void testTwoLayerB()
    {
        final DenseMatrix64F trainingInput = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final DenseMatrix64F expected = new DenseMatrix64F(1,1,true, new double[] {0.4});


        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();

        //build first layer
        final double[] weights2 = {0.3, 0.4};
        builder.setNeuron(new MNeuron(phi, weights2))
                .setReceptiveFieldSize(2 * 2)
                .set1DInputSize(4);

        OldFeatureMap subsamplingMap = new OldSubSamplingMap(builder);

        assertThat(subsamplingMap.getFeatureMap().numCols, is(2));
        assertThat(subsamplingMap.getFeatureMap().numRows, is(2));

        //build second layer
        final double[] weights = {0.01, 0.02, 0.03, 0.04, 0.05};
        builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights))
                .setReceptiveFieldSize(2 * 2)
                .set1DInputSize(2);

        OldFeatureMap convolutionMap = new OldConvolutionMapLayer(builder);

        assertThat(convolutionMap.getFeatureMap().numCols, is(1));
        assertThat(convolutionMap.getFeatureMap().numRows, is(1));

        //build network
        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(subsamplingMap, convolutionMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        OldConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{trainingInput, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);

        final DenseMatrix64F mInducedField0 = network.getLayer(0).mInducedLocalField;
        final DenseMatrix64F mInducedField1 = network.getLayer(1).mInducedLocalField;
        final DenseMatrix64F mImpulseFunc1 = network.getLayer(1).mImpulseFunction;
        final DenseMatrix64F mGradients0 = network.getLayer(0).mGradients;
        final DenseMatrix64F mGradients1 = network.getLayer(1).mGradients;

        //calculate gradient at (0,0) in second layer (1)
        //final double d00 = (expected.get(0,0) - mImpulseFunction.get(0,0)) * phi.derivative( mInducedField.get(0,0) );
        final double d001 = (expected.get(0,0) - mImpulseFunc1.get(0,0)) * phi.derivative( mInducedField1.get(0,0));
        assertThat(d001, is(mGradients1.get(0,0)));

        //calculate gradient at (0,0) in first layer (0)

        final double d00 = phi.derivative(mInducedField0.get(0,0)) * mGradients1.get(0,0) * weights[0];

        System.out.println(toString(d00));
        assertThat(toString(d00), is(toString(mGradients0.get(0, 0))));
    }

    private String toString(double m)
    {
        return String.format("%8.8g", m);
    }

    @Test
    public void testSaveWeightAdjustmentsA()
    {
        final DenseMatrix64F trainingInput = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final DenseMatrix64F expected = new DenseMatrix64F(2,2,true, new double[] {
                0.4, 0.6
                ,0.1, 0.8
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);

        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                .setLayers(featureMap)
                .setLearningParam(0.05)
                .setMomentumParam(0.01);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        OldConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{trainingInput, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);
        backPropagation.saveWeightAdjustments();

        final DenseMatrix64F mImpulseFunction = network.getLayer(0).mImpulseFunction;
        final DenseMatrix64F mInducedField = network.getLayer(0).mInducedLocalField;
        final DenseMatrix64F mGradient = network.getLayer(0).mGradients;
        final double[] aAdjustments = network.getLayer(0).aWeightAdjustments;

        final double w0 = 0.05 *
                (mGradient.get(0,0) * trainingInput.get(0,0)
                + mGradient.get(0,1) * trainingInput.get(0,1)
                + mGradient.get(1,0) * trainingInput.get(1,0)
                + mGradient.get(1,1) * trainingInput.get(1,1));

        assertThat(w0, is(aAdjustments[0]));

        final double w1 = 0.05 *
                (mGradient.get(0,0) * trainingInput.get(0,1)
                        + mGradient.get(0,1) * trainingInput.get(0,2)
                        + mGradient.get(1,0) * trainingInput.get(1,1)
                        + mGradient.get(1,1) * trainingInput.get(1,2));

        assertThat(w1, is(aAdjustments[1]));

        final double w4 = 0.05 *
                ((mGradient.get(0,0)
                        + mGradient.get(0,1)
                        + mGradient.get(1,0)
                        + mGradient.get(1,1)));
        assertThat(w4, is(aAdjustments[4]));
    }

    @Test
    public void testSaveWeightAdjustmentsB()
    {
        final DenseMatrix64F input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final DenseMatrix64F expected = new DenseMatrix64F(2,2,true, new double[] {
                0.4, 0.6
                ,0.1, 0.8
        });

        final double[] weights = {0.3, 0.4};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2*2);
        builder.set1DInputSize(4);

        OldFeatureMap featureMap = new OldSubSamplingMap(builder);

        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                .setLayers(featureMap)
                .setLearningParam(0.05)
                .setMomentumParam(0.0);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        OldConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{input, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);
        backPropagation.saveWeightAdjustments();

        final DenseMatrix64F mGradient = network.getLayer(0).mGradients;
        final double[] aAdjustments = network.getLayer(0).aWeightAdjustments;

        final double w0 = 0.05 *
                         (mGradient.get(0,0) * (input.get(0,0) + input.get(0,1) + input.get(1,0) + input.get(1,1))
                        + mGradient.get(0,1) * (input.get(0,2) + input.get(0,3) + input.get(1,2) + input.get(1,3))
                        + mGradient.get(1,0) * (input.get(2,0) + input.get(2,1) + input.get(3,0) + input.get(3,1))
                        + mGradient.get(1,1) * (input.get(2,2) + input.get(2,3) + input.get(3,2) + input.get(3,3)));

        assertThat(toString(w0), is(toString(aAdjustments[0])));

        final double w1 = 0.05 *
                         (mGradient.get(0,0)
                        + mGradient.get(0,1)
                        + mGradient.get(1,0)
                        + mGradient.get(1,1));

        assertThat(toString(w1), is(toString(aAdjustments[1])));
    }

    @Test
    public void testAdjustWeightsA()
    {
        final DenseMatrix64F trainingInput = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final DenseMatrix64F expected = new DenseMatrix64F(2,2,true, new double[] {
                0.4, 0.6
                ,0.1, 0.8
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);

        OldFeatureMap featureMap = new OldConvolutionMapLayer(builder);

        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(featureMap)
                  .setLearningParam(0.05)
                  .setMomentumParam(0.01);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        OldConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{trainingInput, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);
        backPropagation.saveWeightAdjustments();
        backPropagation.adjustWeights();

        final DenseMatrix64F mGradient = network.getLayer(0).mGradients;
        final double[] aAdjustments = network.getLayer(0).aWeightAdjustments;

        final double w0 = 0.05 *
                (mGradient.get(0,0) * trainingInput.get(0,0)
                        + mGradient.get(0,1) * trainingInput.get(0,1)
                        + mGradient.get(1,0) * trainingInput.get(1,0)
                        + mGradient.get(1,1) * trainingInput.get(1,1));

        assertThat(w0, is(aAdjustments[0]));

        final double w1 = 0.05 *
                (mGradient.get(0,0) * trainingInput.get(0,1)
                        + mGradient.get(0,1) * trainingInput.get(0,2)
                        + mGradient.get(1,0) * trainingInput.get(1,1)
                        + mGradient.get(1,1) * trainingInput.get(1,2));

        assertThat(w1, is(aAdjustments[1]));

        final double w4 = 0.05 *
                ((mGradient.get(0,0)
                        + mGradient.get(0,1)
                        + mGradient.get(1,0)
                        + mGradient.get(1,1)));
        assertThat(w4, is(aAdjustments[4]));

        MNeuron neuron = network.getLayer(0).layer.getNeuron(0);

        assertThat(neuron.getWeight(0), is(weights[0]+w0));
        assertThat(neuron.getWeight(4), is(weights[4]+w4));

        final double w0p = neuron.getWeight(0);

        backPropagation.resetWeightAdjustments();
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);
        backPropagation.saveWeightAdjustments();
        backPropagation.adjustWeights();

        final double w02 = 0.05 *
                (mGradient.get(0,0) * trainingInput.get(0,0)
                        + mGradient.get(0,1) * trainingInput.get(0,1)
                        + mGradient.get(1,0) * trainingInput.get(1,0)
                        + mGradient.get(1,1) * trainingInput.get(1,1));

        assertThat(neuron.getWeight(0), is(w0p + weights[0]*0.01 + w02));
    }

    @Test
    public void testBackPropDoubleLayer()
    {
        Random random = new Random(10001);

        final DenseMatrix64F trainingInput = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final DenseMatrix64F expected = new DenseMatrix64F(1,1,true, new double[] {
                0.4
        });

        final double[] weights = {
                random.nextGaussian(),//0.1
                random.nextGaussian(),//0.2
                random.nextGaussian(),//0.3
                random.nextGaussian(),//0.4
                random.nextGaussian()//0.5
        };

        OldFeatureMap.Builder builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);

        OldFeatureMap firstLayer = new OldConvolutionMapLayer(builder);

        builder = new OldFeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, new double[]{random.nextGaussian(), random.nextGaussian()}))
                .setReceptiveFieldSize(2 * 2)
                .set1DInputSize(2);
        OldFeatureMap secondLayer = new OldSubSamplingMap(builder);

        OldConvolutionalNetwork.Builder netBuilder = new OldConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(firstLayer, secondLayer)
                  .setLearningParam(0.001)
                  .setMomentumParam(0.0000001)
                  .setIterations(10000);

        OldConvolutionalNetwork network = new OldConvolutionalNetwork(netBuilder);
        network.backpropagation(0.001, new DenseMatrix64F[]{trainingInput, expected});

    }
}

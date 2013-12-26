package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

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

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);

        FeatureMap featureMap = new ConvolutionMap(builder);

        ConvolutionalNetwork.Builder netBuilder = new ConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(featureMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        ConvolutionalNetwork network = new ConvolutionalNetwork(netBuilder);
        ConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

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

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2*2);
        builder.set1DInputSize(4);

        FeatureMap featureMap = new SubSamplingMap(builder);

        ConvolutionalNetwork.Builder netBuilder = new ConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(featureMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        ConvolutionalNetwork network = new ConvolutionalNetwork(netBuilder);
        ConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

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
        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(3 * 3);
        builder.set1DInputSize(4);

        FeatureMap convolutionMap = new ConvolutionMap(builder);

        assertThat(convolutionMap.getFeatureMap().numCols, is(2));
        assertThat(convolutionMap.getFeatureMap().numRows, is(2));

        //build second layer
        final double[] weights2 = {0.3, 0.4};
        builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights2));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(2);

        FeatureMap subsamplingMap = new SubSamplingMap(builder);

        assertThat(subsamplingMap.getFeatureMap().numCols, is(1));
        assertThat(subsamplingMap.getFeatureMap().numRows, is(1));

        //build network
        ConvolutionalNetwork.Builder netBuilder = new ConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(convolutionMap, subsamplingMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        ConvolutionalNetwork network = new ConvolutionalNetwork(netBuilder);
        ConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

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


        FeatureMap.Builder builder = new FeatureMap.Builder();

        //build first layer
        final double[] weights2 = {0.3, 0.4};
        builder.setNeuron(new MNeuron(phi, weights2));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(4);

        FeatureMap subsamplingMap = new SubSamplingMap(builder);

        assertThat(subsamplingMap.getFeatureMap().numCols, is(2));
        assertThat(subsamplingMap.getFeatureMap().numRows, is(2));

        //build second layer
        final double[] weights = {0.01, 0.02, 0.03, 0.04, 0.05};
        builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(2);

        FeatureMap convolutionMap = new ConvolutionMap(builder);

        assertThat(convolutionMap.getFeatureMap().numCols, is(1));
        assertThat(convolutionMap.getFeatureMap().numRows, is(1));

        //build network
        ConvolutionalNetwork.Builder netBuilder = new ConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(subsamplingMap, convolutionMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        ConvolutionalNetwork network = new ConvolutionalNetwork(netBuilder);
        ConvolutionalNetwork.BackPropagation backPropagation = network.new BackPropagation();

        network.setupExampleInfo(new DenseMatrix64F[]{trainingInput, expected});
        network.forwardPropagation.calculateForwardPropOnePass(0);
        backPropagation.constructGradients(0);

        //calculate gradient at (0,0) in first layer (0)
        final DenseMatrix64F mInducedField0 = network.getLayer(0).mInducedLocalField;
        final DenseMatrix64F mGradients0 = network.getLayer(0).mGradients;
        final DenseMatrix64F mGradients1 = network.getLayer(1).mGradients;

        final double d00 = phi.derivative(mInducedField0.get(0,0)) * mGradients1.get(0,0) * weights2[0];

        assertThat(d00, is(mGradients0.get(0, 0)));
    }
}

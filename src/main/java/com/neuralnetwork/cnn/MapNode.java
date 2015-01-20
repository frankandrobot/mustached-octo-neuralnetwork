package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.ICnnMap;
import org.ejml.data.DenseMatrix64F;

import java.util.LinkedList;
import java.util.List;

/**
 * Wrapper for the {@link ICnnMap} class that turns it into a (graph) node
 */
class MapNode
{
    /**
     * could be null in the case of input maps
     */
    ICnnMap map;

    DenseMatrix64F yOutput;

    private List<MapNode> lParents = new LinkedList<MapNode>();
    private List<MapNode> lChildren = new LinkedList<MapNode>();
    private DenseMatrix64F[] aInputs;


    public MapNode(ICnnMap map)
    {
        if (map != null)
        {
            this.map = map;
            yOutput = new DenseMatrix64F(map.getOutputDim().rows, map.getOutputDim().cols);
        }
    }

    public MapNode(DenseMatrix64F input)
    {
        yOutput = input;
    }

    public void addInput(MapNode inputMap)
    {
        lParents.add(inputMap);

        aInputs = new DenseMatrix64F[lParents.size()];
    }

    /**
     * Valid only if has map
     */
    public void generateOutput()
    {
        if (aInputs != null)
        {
            this.yOutput = map.generateOutput(getInputs());
        }
    }

    public List<MapNode> getParents()
    {
        return lParents;
    }

    /**
     * Construct the input maps from the parents
     *
     * @return
     */
    private List<ICnnMap> getInputMaps()
    {
        List<ICnnMap> maps = new LinkedList<ICnnMap>();

        for(MapNode mapNode : lParents)
        {
            maps.add(mapNode.map);
        }

        return maps;
    }

    /**
     * Get the matrices containing the input values
     *
     * @return
     */
    private DenseMatrix64F[] getInputs()
    {
        int len = 0;

        for(MapNode map: lParents)
        {
            aInputs[len++] = map.yOutput;
        }

        return aInputs;
    }

    public void validateInputs()
    {
        map.validateInputs(getInputMaps());
    }
}

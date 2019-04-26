from py2neo import Node, Relationship, Graph
import json
import numpy




#################### reading from neo4j database #######################

class Backend:
    graph = None
    def connect(self):
        Backend.graph = Graph(host="canopus.cc.gatech.edu",password='1234')

    def getdata_st(self,sub,roi1,roi2):
        """
        Args:
            sub (int): Subject ID.
            roi1 (str): First ROI name.
            roi2 (str):Second ROI name.

        Returns:
            Object: The return value. True for success, False otherwise.
        """
        query = f'''match (n:ROI)-[r:NOS]->(m:ROI)
        where n.name="{roi1}" and m.name="{roi2}" and r.SUBJECT={sub}
        return n.name as n1,m.name as n2,r._length as _length,r._weight as _weight,r.border as border'''
        A = Backend.graph.run(query).data()
        return A[0]

    def getmania_st(self,sub,roi1,roi2):
        """
        Args:
            sub (int): Subject ID.
            roi1 (str): First ROI name.
            roi2 (str):Second ROI name.

        Returns:
            Object: MANIA2 results.
        """
        query = f'''match (n:ROI)-[r:MANIA2]->(m:ROI)
        where n.name="{roi1}" and m.name="{roi2}" and r.SUBJECT={sub}
        return r.correction_type as correction_type,r.is_connected as is_connected,
        r.is_connected_mania1 as is_connected_mania1, r.threshold1 as threshold1, r.threshold2 as threshold2,
        r.corrected_weights as corrected_weights'''
        A = Backend.graph.run(query).data()
        return A[0]


    def getdata_sts(self,sub,rois,ih = False):
        if ih:
            l = len(rois)//2
            rois=sorted(list(rois))
            roi_list1 = "['" + "','".join(rois[0:l]) + "']"
            roi_list2 = "['" + "','".join(rois[l:]) + "']"
            query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                        WHERE
                        (
                        (n.name IN {roi_list1} AND m.name IN {roi_list2}) OR
                        (n.name IN {roi_list2} AND m.name IN {roi_list1})
                        ) AND
                        r.SUBJECT={sub}
                        RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
        else:
            roi_list = "['" + "','".join(rois) + "']"
            query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                        WHERE n.name IN {roi_list} AND m.name IN {roi_list} AND r.SUBJECT={sub}
                        RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
        out = Backend.graph.run(query).data()
        return out


    def getdata_st_subs(st,subs):
        sub_list = "[" + ','.join(map(str, subs)) + "]"
        query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                    WHERE n.name='{st.roi1}' AND m.name='{st.roi2}' AND r.SUBJECT IN {sub_list}
                    RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
        out = Backend.graph.run(query).data()
        return out


    def getdata_pair_subs(self,st,subs):
        sub_list = "[" + ','.join(map(str, subs)) + "]"
        query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                    WHERE n.name='{st.roi1}' AND m.name='{st.roi2}' AND r.SUBJECT IN {sub_list}
                    RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
        out = Backend.graph.run(query).data()
        query = f'''MATCH (n:ROI)-[r:NOS]->(m:ROI)
                    WHERE n.name='{st.roi2}' AND m.name='{st.roi1}' AND r.SUBJECT IN {sub_list}
                    RETURN n.name as n1, m.name as n2, r._length as _length, r._weight as _weight, r.border as border'''
        out.extend(Backend.graph.run(query).data())
        return out


    def get_roi_regressor(self,roi, subject=None):
        """Get the ROI regressor for a ROI and a subject (optional)

        Args:
            roi (str): ROI to which the ROI
            subject (int or str): Subject ID for which ROI regressor is needed. If None (default), ROI regressor
                                for all subjects is retured.

        Returns:
            dict: ROI regressor for the subject. If subject is None, ROI regressor for all subjects
        """
        roi_regressor = Backend.graph.run(f"MATCH (n:ROI{{name:'{roi}'}}) RETURN n.roi_regressor").evaluate()
        if roi_regressor is None or roi_regressor == '':
            roi_regressor = '{}'
        roi_regressor_dict = json.loads(roi_regressor)
        if subject is None:
            return roi_regressor_dict
        else:
            return roi_regressor_dict[str(subject)]


    #################### writing to neo4j database #######################
    def write_connection(self,roi1, roi2, relation_type, attributes, run=True):
        """Write a connection to Neo4j database

        Args:
            roi1 (str): Source ROI
            roi2 (str): Destination ROI
            relation_type (str): Type of relationship to update
            attributes (dict): Dictionary of attributes to update
            run (bool): Whether to run the query or return the query

        Returns:
            str: Query generated if run is False
        """
        match = f"MATCH (n:ROI{{name:'{roi1}'}}), (m:ROI{{name:'{roi2}'}}) "
        attribs = f"CREATE (n)-[:{relation_type} {{"
        for key in attributes:
            if type(attributes[key]) == str:
                attribs += str(key) + ":'" + str(attributes[key]) + "', "
            elif type(attributes[key]) in [list, numpy.ndarray]:
                attribs += str(key) + ":[" + ','.join([str(x) for x in attributes[key]]) + "], "
            else:
                attribs += str(key) + ':' + str(attributes[key]) + ', '
        query = match + attribs[:-2] + '}]->(m)'
        if run:
            Backend.graph.run(query)
        else:
            return query


    def update_connection(self,roi1, roi2, relation_type, subject, attributes, run=True):
        """Update a connection in Neo4j database

        Args:
            roi1 (str): Source ROI
            roi2 (str): Destination ROI
            relation_type (str): Type of relationship to update
            subject (int or str): Subject ID
            attributes (dict): Dictionary of attributes to update
            run (bool): Whether to run the query or return the query

        Returns:
            str: Query generated if run is False
        """
        if type(subject) != str:
            subject = str(subject)
        match = f"MATCH (n:ROI{{name:'{roi1}'}})-[r:{relation_type}]->(m:ROI{{name:'{roi2}'}}) WHERE r.SUBJECT={subject}"
        attribs = ' SET '
        for key in attributes:
            if type(attributes[key]) == str:
                attribs += 'r.' + str(key) + "='" + str(attributes[key]) + "', "
            elif type(attributes[key]) == list:
                attribs += 'r.' + str(key) + "=[" + ','.join([str(x) for x in attributes[key]]) + "], "
            else:
                attribs += 'r.' + str(key) + '=' + str(attributes[key]) + ', '
        query = match + attribs[:-2]
        if run:
            Backend.graph.run(query)
        else:
            return query


    def update_roi_regressor(self,subject, run=True):
        """Add or update the ROI Regressor for the subject in Neo4j database

        Args:
            subject (pymania.base.EnsembleST): Subject for which the ROI Regressor has to be added or updated
            run (bool): Whether to run the query or return the list of queries

        Returns:
            list: List of queries to be run
        """
        queries = []
        for roi in subject.rois:
            roi_regressor = Backend.graph.run(f"MATCH (n:ROI{{name:'{roi}'}}) RETURN n.roi_regressor").evaluate()
            if roi_regressor is None or roi_regressor == '':
                roi_regressor = '{}'
            roi_regressor_dict = json.loads(roi_regressor)
            roi_regressor_dict[str(subject.subject)] = {'s': subject.roi_regressors['s-'+roi].to_list(),
                                                        't': subject.roi_regressors['s-'+roi].to_list()}
            roi_regressor_json = json.dumps(roi_regressor_dict)
            query = f"MATCH (n:ROI{{name:'{roi}'}}) SET n.roi_regressor='{roi_regressor_json}'"
            if run:
                Backend.graph.run(query)
            else:
                queries.append(query)
        if not run:
            return queries

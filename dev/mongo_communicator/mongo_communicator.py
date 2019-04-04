from datetime import datetime, timedelta
import time
import threading
import logging

from pymongo import MongoClient, ReturnDocument

logger = logging.getLogger(__name__)


class MongoCommunicator(object):
    """A communicator for distributed running of DeepArchitect that is based on
    a MongoDB database. The communicator is used to implement a master/worker
    paradigm where the master pushes jobs to a work queue which are consumed by
    the workers. The workers then send back results to the master. The
    communicator is structured so that it can only process one job at a time per
    subscription.
    """

    INTERMEDIATE_SUFFIX = '__internal__intermediate'

    def __init__(self,
                 host='localhost',
                 port=27017,
                 refresh_period=180,
                 data_refresher=False):
        """Constructs the MongoDB based communicator.
        Arguments:
            - host: The host where the MongoDB server is running
            - port: The port where the MongoDB server is running
            - refresh_period: The period of time in seconds between refreshing
                data stored in the intermediate tables. If there is data in the
                intermediate tables that has not been refreshed in this period,
                it is moved back to the original table.
        """
        self._client = MongoClient(host, port)
        self._db = self._client['deep-architect']
        self._subscribed = {}
        self._processing = {}
        self._store_intermediate = {}
        self._refresh_period = refresh_period
        self._refresh_thread = None
        self._is_running = True
        if data_refresher:
            self._refresh_thread = threading.Thread(
                target=self._move_stale_data)
            self._refresh_thread.start()

    def __del__(self):
        self._is_running = False
        if self._refresh_thread is not None:
            self._refresh_thread.join()

    def _move_stale_data(self):
        while self._is_running:
            logger.info('Moving stale data in DB')
            for collection in self._db.list_collection_names():
                if MongoCommunicator.INTERMEDIATE_SUFFIX in collection:
                    intermediate_collection = self._db[collection]
                    queue_collection = self._db[collection[:-(
                        len(MongoCommunicator.INTERMEDIATE_SUFFIX))]]
                    query = {
                        'last_refreshed': {
                            '$lt':
                            datetime.now() -
                            timedelta(seconds=self._refresh_period + 1)
                        }
                    }
                    data = intermediate_collection.find_one_and_delete(query)
                    while data is not None:
                        logger.info(
                            'Moving data from %s to %s: %s', collection,
                            collection[:-(
                                len(MongoCommunicator.INTERMEDIATE_SUFFIX))],
                            str(data))
                        queue_collection.insert_one(data)
                        data = intermediate_collection.find_one_and_delete(
                            query)
            time.sleep(self._refresh_period)

    def publish(self, topic, data):
        """Publishes data to some topic. Blocking call. Data is put under 'data'
            key.
        Arguments:
            - topic: The topic to publish the data to
            - data: bson compatible object with the data to publish to topic
        """
        collection = self._db[topic]
        collection.insert_one({'data': data})

    def subscribe(self, subscription, callback, store_intermediate=True):
        """Subscribes to some topic. Non-blocking call. If store_intermediate is
        True, then after each message is consumed and finished processing,
        finish_processing must be called with the original message.
        Arguments:
            - subscription: The name of the topic to subscribe to.
            - callback: Function that is called with dictionary where the object
                data is under the 'data' key.
            - store_intermediate: This parameter controls whether intermediate
                job configs are stored while they are being processed
        """
        if subscription in self._subscribed:
            raise RuntimeError('Already subscribed to this subscription')
        logger.info('Subscribed to %s', subscription)
        self._subscribed[subscription] = True
        self._store_intermediate[subscription] = store_intermediate
        thread = threading.Thread(
            target=self._subscribe, args=(subscription, callback))
        thread.start()

    def _subscribe(self, subscription, callback):
        collection = self._db[subscription]
        self._processing[subscription] = False

        while self._subscribed[subscription]:
            # The current job is still being processed
            if self._processing[subscription]:
                time.sleep(10)
                continue

            data = collection.find_one_and_delete({'_id': {'$exists': True}})
            # Nothing currently in the subscription queue
            if data is None:
                time.sleep(10)
            else:
                self._processing[subscription] = True
                if self._store_intermediate[subscription]:
                    intermediate_collection = self._db[
                        subscription + MongoCommunicator.INTERMEDIATE_SUFFIX]
                    data['last_refreshed'] = datetime.now()
                    intermediate_collection.insert_one(data)

                    def refresh_data(data):
                        time.sleep(self._refresh_period)
                        while intermediate_collection.find_one_and_update(
                            {
                                '_id': data['_id']
                            }, {'$currentDate': {
                                'last_refreshed': True
                            }},
                                return_document=ReturnDocument.AFTER):
                            logger.info('Refreshed data for id %s',
                                        str(data['_id']))
                            time.sleep(self._refresh_period)

                    refresh_thread = threading.Thread(
                        target=refresh_data, args=(data,))
                    refresh_thread.start()
                callback(data)

    def finish_processing(self, subscription, data, success=True):
        """Removes the message from the intermediate processing storage. Must be
        called for every message received if store_intermediate is True.
        Arguments:
            - subscription: The name of the topic to subscribe to.
            - callback: Function that is called with the object representing
                the data that was consumed.
            - success: whether the processing of the message was successful. If
                not, the data is put back into the original queue
        """
        logger.info('Finish processing %s', str(data))
        if self._store_intermediate[subscription]:
            collection = self._db[subscription +
                                  MongoCommunicator.INTERMEDIATE_SUFFIX]
            collection.find_one_and_delete({'_id': data['_id']})
        if not success:
            collection = self._db[subscription]
            collection.insert_one(data)
        self._processing[subscription] = False

    def unsubscribe(self, subscription):
        """Stops communicator from listening to subscription:
            - subscription: The name of the topic to unsubscribe from.
        """
        logger.info('Unsubscribed from %s', subscription)
        self._subscribed[subscription] = False

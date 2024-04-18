import gi
from chat import *
gi.require_version("Gtk", "3.0")

from gi.repository import Gtk

expression = {'Bonjour':'Bonjour. Comment allez vous','Comment allez vous':'Je vais bien et vous','Je vais bien et vous':'Tres bien merci'}

def response(msg):
    intents= predict_class(msg)
    res = get_response(msg, intents)
    return "Ouistiti (°-°) "+res

@Gtk.Template(filename="ui/ui.glade")
class Chat(Gtk.Window):
    __gtype_name__ = "main"

    chatEntry: Gtk.Entry = Gtk.Template.Child()
    receivedStream: Gtk.Box = Gtk.Template.Child()
    sendStream: Gtk.Box = Gtk.Template.Child()
    

    @Gtk.Template.Callback()
    def onDestroy(self, *args):
        Gtk.main_quit()

    @Gtk.Template.Callback()
    def onSend(self, button):
        '''
            Cette fonction est appelé a l'appui du button
            self.chatEntry.get_text() represente le text entrer
        '''
        self.sendMessage(self.chatEntry.get_text())
        self.chatEntry.set_text('')
        self.receiveMessage(response(self.chatEntry.get_text()))

    def sendMessage(self,message):
        message = Gtk.Label(label=message)
        self.sendStream.pack_start(message,False,False,0)
        Chat.Separate(self.receivedStream)
        message.set_visible(True)
        

    def receiveMessage(self,message):
        message = Gtk.Label(label=message)
        self.receivedStream.pack_start(message,False,False,0)
        Chat.Separate(self.sendStream)
        message.set_visible(True)
    
    def Separate(boxStream):
        separator = Gtk.Label(label=' ')
        boxStream.pack_start(separator,False,False,0)
        separator.set_visible(True)

window = Chat()
window.show()

Gtk.main()

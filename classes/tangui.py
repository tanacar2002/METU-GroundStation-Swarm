import time
try:
    import pygame
except:
    print("Pygame should be installed. pip install pygame")
    raise ImportError

class ProgressBar:
    inner_color = (200,220,210)
    progress_color = (0,255,0)
    text_color = (255,255,255)
    border_color = (128,128,128)
    border_width = 3
    border_radius = 5
    deriv,eta = 0,0
    eta_visible = False

    def __init__(self,x=0,y=0,width=300,height=25,min=0,max=100,value=0):
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0],int(height))
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min = min
        self.max = max
        self.valuerange = abs(self.max - self.min)
        self.value = value
        self.resetTime()
        self.text = self.font.render("%{:4.1f}".format(self.value*100/self.valuerange), True, self.text_color)
        self.textrect = self.text.get_size()
        self.etatext = self.font.render("ETA: {:d}m {:2.0f}s".format(int(self.eta//60),self.eta%60), True, self.text_color)
        self.etatextrect = self.etatext.get_size()

    def resetTime(self):
        self.reftime = time.time()
        self.refvalue = self.value

    def toggleEta(self,*args):
        if len(args) == 1:
            self.eta_visible = args[0]
        else:
            self.eta_visible = not self.eta_visible
        self.resetTime()

    def setColor(self,color):
        self.inner_color = color

    def posScale(self,scalex,scaley):
        self.x = scalex*self.x
        self.y = scaley*self.y
        self.width = scalex*self.width
        self.height = scaley*self.height
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0],int(self.height))
        self.text = self.font.render("%{:4.1f}".format(self.value*100/self.valuerange), True, self.text_color)
        self.textrect = self.text.get_size()
        self.etatext = self.font.render("ETA: {:d}m {:2.0f}s".format(int(self.eta//60),self.eta%60), True, self.text_color)
        self.etatextrect = self.etatext.get_size()

    def setValue(self,value):
        self.value = value
        try:
            self.deriv = (self.value-self.refvalue)/(time.time()-self.reftime)
            self.eta = (self.max-self.value)/self.deriv
        except:
            self.eta = 0
        self.text = self.font.render("%{:4.1f}".format(self.value*100/self.valuerange), True, self.text_color)
        self.textrect = self.text.get_size()
        if self.eta < 0:
            self.etatext = self.font.render("Completed!", True, self.text_color)
        else:
            self.etatext = self.font.render("ETA: {:d}m {:2.0f}s".format(int(self.eta//60),self.eta%60), True, self.text_color)
        self.etatextrect = self.etatext.get_size()

    def draw(self,surface):
        pygame.draw.rect(surface,self.inner_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),border_radius=self.border_radius)
        pygame.draw.rect(surface,self.progress_color,(self.x-self.width//2+self.border_width,self.y-self.height//2,(self.width-self.border_width*2)*self.value/self.valuerange,self.height),border_radius=self.border_radius)
        pygame.draw.rect(surface,self.border_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),width=self.border_width,border_radius=self.border_radius)
        surface.blit(self.text,(self.x + self.width//2 + self.border_width, self.y - self.textrect[1]//2))
        if self.eta_visible:
            surface.blit(self.etatext,(self.x - self.etatextrect[0]//2, self.y - self.etatextrect[1]//2 + self.height))

class Selector:
    inner_color = (255,255,255)
    text_color = (0,0,0)
    border_color = (128,128,128)
    border_width = 2
    border_radius = 0
    index = 0

    def __init__(self,x=0,y=0,width=50,height=20,choices=["Choice"]):
        self.font = pygame.font.SysFont(
            pygame.font.get_fonts()[0], int(height))  # int(height))
        self.x = x
        self.y = y
        self.height = height*(30/20)
        self.values = choices
        self.choices = []
        for i,choice in enumerate(self.values):
            self.choices.append(Button(i,x = self.x,height=self.height,text=str(choice)))
        self.width = max(max([choice.width for choice in self.choices]),width)
        for i,choice in enumerate(self.choices):
            choice.width = self.width
            choice.height = self.height
            choice.y = self.y + self.height*(i+1)
            choice.visibility = False
            choice.border_radius = 0

    def getValue(self):
        return self.values[self.index]

    def getChoice(self):
        return self.current_choice

    def updateChoices(self,choices):
        self.__init__(self.x, self.y, self.width,
                      self.height*(20/30), choices=choices)
        if self.index >= len(choices):
            self.index = 0

    def checkHover(self,pos):
        for choice in self.choices:
            if choice.checkHover(pos) and choice.visibility:
                self.index = int(choice.clickFunction())
                self.clickFunction()

        return (self.x-self.width//2 < pos[0] < self.x + self.width//2) and (self.y - self.height//2 < pos[1] < self.y + self.height//2)

    def setColor(self,color):
        self.inner_color = color

    def posScale(self,scalex,scaley):
        self.__init__(self.x*scalex,self.y*scaley,scalex*self.width,scaley*self.height,self.values)
        # self.x = scale*self.x
        # self.y = scale*self.y
        # self.width = scale*self.width
        # self.height = scale*self.height
        # for choice in self.choices:
        #     choice.x = scale*choice.x
        #     choice.y = scale*choice.y
        #     choice.width = scale*choice.width
        #     choice.height = scale*choice.height

    def clickFunction(self):
        for choice in self.choices:
            choice.toggle()

    def draw(self,surface):
        for choice in self.choices:
            choice.draw(surface)
        self.text = self.font.render(str(self.values[self.index]), True, self.text_color)
        self.textrect = self.text.get_size()
        pygame.draw.rect(surface,self.inner_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),border_radius=self.border_radius)
        pygame.draw.rect(surface,self.border_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),width=self.border_width,border_radius=self.border_radius)
        surface.blit(self.text,(self.x - self.textrect[0]//2, self.y - self.textrect[1]//2))

    def __del__(self):
        for choice in self.choices:
            del choice

class Text:
    def __init__(self,x,y,text,font,color=(255,255,255),allign='center'):
        self.font = font
        self.textstr = text
        self.color = color
        self.text = self.font.render(self.textstr, True, self.color)
        self.x = x
        self.y = y
        self.allignment = allign

    def setText(self,text):
        self.textstr = text
        self.text = self.font.render(self.textstr, True, self.color)

    def posScale(self,scalex,scaley):
        self.x = scalex*self.x
        self.y = scaley*self.y

    def setAllignment(self,allign):
        self.allignment = allign

    def draw(self,surface):
        surface.blit(self.text,(self.x-(self.text.get_width()//2 if self.allignment.lower() == 'center' else 0),self.y-(self.text.get_height()//2 if self.allignment.lower() == 'center' else 0)))

    def __str__(self):
        return self.textstr

class Button:
    inner_color = (255,255,255)
    text_color = (0,0,0)
    border_color = (128,128,128)
    border_width = 2
    border_radius = 5
    visibility = True

    def function(self): pass

    def __init__(self,id,x=0,y=0,width=50,height=30,text="Button"):
        self.id = str(id)
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0],int(height*(20/30)))
        self.x = x
        self.y = y
        self.textstr = str(text)
        self.text = self.font.render(self.textstr, True, self.text_color)
        self.textrect = self.text.get_size()
        self.width = max(width,self.textrect[0] + self.border_width*3)
        self.height = max(height,self.textrect[1] + self.border_width*3)

    def getText(self):
        return self.textstr

    def checkHover(self,pos):
        return (self.x-self.width//2 < pos[0] < self.x + self.width//2) and (self.y - self.height//2 < pos[1] < self.y + self.height//2)

    def setFunction(self,func):
        self.function = func

    def toggle(self):
        self.visibility = not self.visibility

    def posScale(self,scalex,scaley):
        self.x = scalex*self.x
        self.y = scaley*self.y
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0], int(scaley*self.height*(20/30)))

    def setColor(self,color):
        self.inner_color = color

    def clickFunction(self):
        if self.visibility:
            self.function()
            return self.id

    def draw(self,surface):
        if self.visibility:
            pygame.draw.rect(surface,self.inner_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),border_radius=self.border_radius)
            pygame.draw.rect(surface,self.border_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),width=self.border_width,border_radius=self.border_radius)
            surface.blit(self.text,(self.x - self.textrect[0]//2, self.y - self.textrect[1]//2))

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #fff
}
.chat-message.bot {
    background-color: #FFF
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #000;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://global.discourse-cdn.com/openai1/original/4X/0/3/2/032099ce2a0d9e32db3e106d303713eee2ce3df5.webp" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://images.pexels.com/photos/614810/pexels-photo-614810.jpeg?cs=srgb&dl=pexels-simon-robben-614810.jpg&fm=jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
